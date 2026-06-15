"""
runners/build_student.py
────────────────────────
Phase 2 / ADR-0012 — the GENERIC student builder. The system constructs a VLM
from a declarative StudentSpec; the human writes this harness once, and every
*instance* of construction is a spec the Search Strategist proposes.

Pipeline (per StudentSpec):
    assemble  → wire vision encoder + projector + LM into one StudentVLM
    align     → stage-1: train ONLY the projector to connect the modalities
    distill   → stage-2: LoRA-distill the assembled student from the teacher cache
    evaluate  → same-path MCQ eval (P2-1.3 methodology)   [full wiring: B1.3]
    record    → MetricsReport-shaped record keyed by spec.content_hash()

B1.0 scope: this is the SKELETON + an end-to-end SMOKE (`--smoke`) that proves the
pipeline wires together on the 16GB Mac (assemble + 2-step align + 2-step distill +
a few greedy generations). The full VLMEvalKit eval integration is B1.3; until then
`evaluate()` runs a generation sanity check and marks the run accordingly.

Usage
-----
    # end-to-end smoke (tiny, ~minutes) — proves the build wires together
    python runners/build_student.py --spec tests/fixtures/student_spec_p2b1_qwen05b_siglip.json --smoke

    # real construction run (compute-gated, multi-hour) — B1.3
    python runners/build_student.py --spec <spec.json> --out artifacts/students/<name>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.students import StudentSpec  # noqa: E402

# Named data keys → (cache JSONL, image dir). The agent selects a key in the spec;
# the human maintains this registry as caches are produced (B1.1 adds qa_balanced).
DATA_REGISTRY: dict[str, tuple[str, str]] = {
    "coco_caption_5k": ("datasets/caption_cache/qwen25_3b_coco5k.jsonl", "datasets/coco_train2017"),
    "qa_5k":           ("datasets/caption_cache/qwen25_3b_qa5k.jsonl",   "datasets/coco_train2017"),
    # B1.1 balanced hard-negative recipe exists (services/distillation_pipeline.py
    # --mode qa_balanced). The full balanced cache is a compute-gated B1.3 step;
    # until generated, this key falls back to the plain qa5k cache so the smoke runs.
    "qa_balanced_5k":  ("datasets/caption_cache/qwen25_3b_qa_balanced5k.jsonl", "datasets/coco_train2017"),
    "qa_balanced_5k_fallback": ("datasets/caption_cache/qwen25_3b_qa5k.jsonl",  "datasets/coco_train2017"),
    "canary":          ("datasets/caption_cache/canary.jsonl",          "datasets/coco_train2017"),
}

CAPTION_PROMPT = "Describe this image in detail."


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _resolve_data(key: str) -> tuple[Path, Path]:
    if key not in DATA_REGISTRY:
        raise SystemExit(f"unknown data key '{key}' — known: {sorted(DATA_REGISTRY)}")
    cache, images = DATA_REGISTRY[key]
    cache_p, images_p = PROJECT_ROOT / cache, PROJECT_ROOT / images
    # If the primary cache isn't generated yet but a '<key>_fallback' exists, use it.
    if not cache_p.exists() and f"{key}_fallback" in DATA_REGISTRY:
        fb_cache, fb_images = DATA_REGISTRY[f"{key}_fallback"]
        fb_cache_p = PROJECT_ROOT / fb_cache
        if fb_cache_p.exists():
            print(f"  note: '{key}' cache not built yet → falling back to '{key}_fallback'")
            return fb_cache_p, PROJECT_ROOT / fb_images
    return cache_p, images_p


# ── Assembled student ─────────────────────────────────────────────────────────

class _Proc:
    """Tiny holder so StudentVLM.infer() can reach its tokenizer + image processor."""
    def __init__(self, tokenizer, image_processor):
        self.tok = tokenizer
        self.image = image_processor


class StudentVLM(nn.Module):
    """A VLM assembled from a pretrained vision encoder + a fresh MLP projector +
    a pretrained causal LM. Image patch features are projected into the LM
    embedding space and PREPENDED to the text token embeddings (LLaVA-style),
    so no special placeholder token is required in the tokenizer.

    forward(input_ids, attention_mask, pixel_values, labels) → CausalLMOutput.
    """

    def __init__(self, vision: nn.Module, projector: nn.Module, lm: nn.Module):
        super().__init__()
        self.vision = vision
        self.projector = projector
        self.lm = lm

    def _image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.vision(pixel_values=pixel_values).last_hidden_state  # [B, P, vdim]
        return self.projector(feats)                                      # [B, P, H]

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        text_embeds = self.lm.get_input_embeddings()(input_ids)           # [B, T, H]
        img_embeds = self._image_embeds(pixel_values).to(text_embeds.dtype)
        inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)       # [B, P+T, H]

        b, p, _ = img_embeds.shape
        img_mask = torch.ones(b, p, dtype=attention_mask.dtype, device=attention_mask.device)
        attn = torch.cat([img_mask, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            img_labels = torch.full((b, p), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([img_labels, labels], dim=1)

        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=full_labels)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values, **kw):
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        img_embeds = self._image_embeds(pixel_values).to(text_embeds.dtype)
        inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
        b, p, _ = img_embeds.shape
        img_mask = torch.ones(b, p, dtype=attention_mask.dtype, device=attention_mask.device)
        attn = torch.cat([img_mask, attention_mask], dim=1)
        return self.lm.generate(inputs_embeds=inputs_embeds, attention_mask=attn, **kw)

    @torch.no_grad()
    def infer(self, image_path: str, question: str, is_mcq: bool) -> str:
        """Eval interface matching eval_vlmeval's model protocol (same-path scoring)."""
        from runners.eval_vlmeval import MCQ_PROMPT_SUFFIX, POPE_PROMPT_SUFFIX
        device = next(self.lm.parameters()).device
        image = Image.open(image_path).convert("RGB")
        pixel_values = self._proc.image(images=image, return_tensors="pt").pixel_values.to(device)
        text = question + (MCQ_PROMPT_SUFFIX if is_mcq else POPE_PROMPT_SUFFIX)
        msg = [{"role": "user", "content": text}]
        prompt = self._proc.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = self._proc.tok(prompt, return_tensors="pt").to(device)
        out = self.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                            pixel_values=pixel_values, max_new_tokens=32, do_sample=False)
        # generate(inputs_embeds=...) returns ONLY new tokens (no prompt prefix).
        return self._proc.tok.decode(out[0], skip_special_tokens=True).strip()

    # set by assemble()/load_student so infer() can reach the processors
    _proc = None


def assemble(spec: StudentSpec, device: str) -> tuple[StudentVLM, object, object]:
    """Build the StudentVLM from the spec. Returns (model, tokenizer, image_processor)."""
    from transformers import (AutoImageProcessor, AutoModel,
                              AutoModelForCausalLM, AutoTokenizer)

    print(f"  assembling: vision={spec.vision}  lm={spec.lm}  projector={spec.projector}")
    vision_full = AutoModel.from_pretrained(spec.vision, trust_remote_code=True)
    vision = getattr(vision_full, "vision_model", vision_full)  # SigLIP/CLIP → vision tower
    image_processor = AutoImageProcessor.from_pretrained(spec.vision)
    vdim = vision.config.hidden_size

    lm = AutoModelForCausalLM.from_pretrained(
        spec.lm, torch_dtype=torch.float32, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(spec.lm, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    h = lm.config.hidden_size

    # Projector MLP: depth Linear layers (vdim → hidden → … → H), GELU between.
    p = spec.projector
    layers: list[nn.Module] = []
    if p.depth == 1:
        layers.append(nn.Linear(vdim, h))
    else:
        layers.append(nn.Linear(vdim, p.hidden))
        for _ in range(p.depth - 2):
            layers += [nn.GELU(), nn.Linear(p.hidden, p.hidden)]
        layers += [nn.GELU(), nn.Linear(p.hidden, h)]
    projector = nn.Sequential(*layers)

    model = StudentVLM(vision, projector, lm).to(device)
    model._proc = _Proc(tokenizer, image_processor)
    n_proj = sum(x.numel() for x in projector.parameters())
    print(f"  assembled : vdim={vdim} → H={h}, projector params={n_proj:,}")
    return model, tokenizer, image_processor


# ── Data ──────────────────────────────────────────────────────────────────────

def _load_rows(data_key: str, limit: int | None) -> list[dict]:
    """Reuse the finetune cache loader → unified {image_path, prompt, target} rows."""
    from runners.finetune_vlm import load_cache
    cache, images = _resolve_data(data_key)
    if not cache.exists():
        raise SystemExit(f"data '{data_key}' cache missing: {cache}")
    rows = load_cache(cache, images, max_samples=limit, default_prompt=CAPTION_PROMPT)
    if not rows:
        raise SystemExit(f"no usable rows in {cache}")
    return rows


def _encode(row: dict, tokenizer, image_processor, device: str):
    """Batch-1 encode of (image, prompt, target) with prompt tokens masked to -100."""
    image = Image.open(row["image_path"]).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

    prompt_ids = tokenizer(row["prompt"], add_special_tokens=True).input_ids
    target_ids = tokenizer(" " + row["target"], add_special_tokens=False).input_ids
    target_ids = target_ids + [tokenizer.eos_token_id]
    input_ids = torch.tensor([prompt_ids + target_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[0, : len(prompt_ids)] = -100
    return input_ids, attention_mask, pixel_values, labels


def _prompt_inputs(row: dict, tokenizer, image_processor, device: str):
    """Encode the prompt only (for generation sanity)."""
    image = Image.open(row["image_path"]).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
    enc = tokenizer(row["prompt"], return_tensors="pt").to(device)
    return enc.input_ids, enc.attention_mask, pixel_values


# ── Stages ──────────────────────────────────────────────────────────────────

def _train_loop(model, rows, tokenizer, image_processor, device, steps, lr, label):
    """Minimal batch-1 training loop over the given trainable params (already set).

    Polls the operator run control (ADR-0013 H1) between steps: pause blocks here,
    stop ends the stage gracefully (caller still saves), kill aborts the build.
    """
    from services import run_control as rc
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    losses = []
    for i in range(steps):
        try:
            rc.checkpoint()
        except rc.RunStopped as e:
            print(f"    [{label}] operator '{e.mode}' at step {i+1}/{steps}"
                  + (f" ({e.reason})" if e.reason else ""))
            if e.mode == "kill":
                raise
            break  # stop: graceful — return what we have, caller saves
        row = rows[i % len(rows)]
        input_ids, attn, pixel_values, labels = _encode(row, tokenizer, image_processor, device)
        out = model(input_ids=input_ids, attention_mask=attn,
                    pixel_values=pixel_values, labels=labels)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        loss_val = out.loss.detach().item()
        losses.append(loss_val)
        print(f"    [{label}] step {i+1}/{steps}  loss={loss_val:.3f}")
    return losses


def align(model, spec, tokenizer, image_processor, device, steps, lr=1e-3, limit=None):
    """Stage-1: freeze vision + LM, train ONLY the projector."""
    for pm in model.vision.parameters(): pm.requires_grad = False
    for pm in model.lm.parameters(): pm.requires_grad = False
    for pm in model.projector.parameters(): pm.requires_grad = True
    rows = _load_rows(spec.align.data, limit)
    print(f"  align     : projector-only, {steps} steps on '{spec.align.data}' ({len(rows)} rows)")
    return _train_loop(model, rows, tokenizer, image_processor, device, steps, lr, "align")


def distill(model, spec, tokenizer, image_processor, device, steps, lr=2e-4, limit=None):
    """Stage-2: LoRA-adapt the LM, keep training the projector, distill on the teacher cache."""
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(r=spec.distill.lora_r, lora_alpha=2 * spec.distill.lora_r,
                      target_modules=["q_proj", "v_proj", "o_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model.lm = get_peft_model(model.lm, lora)
    for pm in model.projector.parameters(): pm.requires_grad = True  # projector stays trainable
    rows = _load_rows(spec.distill.data, limit)
    print(f"  distill   : LoRA r={spec.distill.lora_r} + projector, {steps} steps on "
          f"'{spec.distill.data}' ({len(rows)} rows)")
    return _train_loop(model, rows, tokenizer, image_processor, device, steps, lr, "distill")


def generate_sanity(model, spec, tokenizer, image_processor, device, n):
    """Greedy-generate on n samples — proves the assembled forward+decode path works."""
    rows = _load_rows(spec.eval.data if hasattr(spec.eval, "data") else spec.distill.data, n)
    model.eval()
    outs = []
    for row in rows[:n]:
        input_ids, attn, pixel_values = _prompt_inputs(row, tokenizer, image_processor, device)
        gen = model.generate(input_ids=input_ids, attention_mask=attn,
                             pixel_values=pixel_values, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        outs.append({"prompt": row["prompt"], "output": text})
        print(f"    [gen] {row['prompt'][:40]!r} → {text[:60]!r}")
    return outs


# ── Persistence (so the trained student can be reloaded for eval) ────────────

def save_student(model: StudentVLM, spec: StudentSpec, out_dir: Path) -> Path:
    """Persist the trained student: projector weights + LoRA adapter + processors + spec.
    Vision + LM base weights are reloaded from HF (spec.lm/spec.vision), so only the
    *learned* parts are stored."""
    sdir = out_dir / "student"
    sdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), sdir / "projector.pt")
    # LoRA adapter (model.lm is a PeftModel after distill)
    if hasattr(model.lm, "save_pretrained"):
        try:
            model.lm.save_pretrained(str(sdir / "lora_adapter"))
        except Exception as exc:
            print(f"  WARN: could not save LoRA adapter: {exc}")
    model._proc.tok.save_pretrained(str(sdir / "processor"))
    model._proc.image.save_pretrained(str(sdir / "processor"))
    (sdir / "spec.json").write_text(spec.model_dump_json(indent=2))
    print(f"  saved student → {sdir}")
    return sdir


def load_student(build_dir: Path, device: str | None = None) -> StudentVLM:
    """Reconstruct a saved StudentVLM (projector weights + LoRA adapter) for eval."""
    from peft import PeftModel
    device = device or _device()
    sdir = build_dir / "student" if (build_dir / "student").exists() else build_dir
    spec = StudentSpec.model_validate_json((sdir / "spec.json").read_text())
    model, tokenizer, image_processor = assemble(spec, device)
    model.projector.load_state_dict(torch.load(sdir / "projector.pt", map_location=device))
    adapter = sdir / "lora_adapter"
    if adapter.exists():
        model.lm = PeftModel.from_pretrained(model.lm, str(adapter)).to(device)
    model._proc = _Proc(tokenizer, image_processor)
    model.eval()
    return model


# ── Orchestration ──────────────────────────────────────────────────────────

def build(spec: StudentSpec, out_dir: Path, smoke: bool,
          align_steps: int | None = None, distill_steps: int | None = None,
          max_samples: int | None = None, save: bool | None = None) -> dict:
    """Run the full pipeline for a spec. Returns a run record (also written to disk).

    Budgets: smoke uses tiny fixed budgets; a real run uses the spec's align.steps and
    a distill-step count derived from epochs × dataset size (overridable via args).
    """
    device = _device()
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_id = spec.content_hash()
    started = datetime.now(timezone.utc)
    print(f"▶ build_student  spec={exp_id[:12]}  device={device}  smoke={smoke}")

    # Operator intake (ADR-0013 H1): stamp the authorized goal/scope if a run.yaml exists.
    run_meta = None
    run_yaml = PROJECT_ROOT / "run.yaml"
    if run_yaml.exists():
        try:
            from schemas.run_config import load_run_config
            run_meta = json.loads(load_run_config(run_yaml).model_dump_json())
            print(f"  run.yaml  : {run_meta['goal'][:70]}")
        except Exception as exc:
            print(f"  WARN: run.yaml present but unreadable: {exc}")

    if smoke:
        align_steps, distill_steps_eff, eval_n, data_limit = 2, 2, 3, 8
    else:
        align_steps = align_steps or spec.align.steps
        data_limit = max_samples
        n_distill_rows = len(_load_rows(spec.distill.data, data_limit))
        distill_steps_eff = distill_steps or (spec.distill.epochs * n_distill_rows)
        eval_n = spec.eval.n

    model, tokenizer, image_processor = assemble(spec, device)
    align_losses = align(model, spec, tokenizer, image_processor, device, align_steps, limit=data_limit)
    distill_losses = distill(model, spec, tokenizer, image_processor, device,
                             distill_steps_eff, limit=data_limit)
    gens = generate_sanity(model, spec, tokenizer, image_processor, device, eval_n if smoke else 3)

    saved_dir = None
    if save if save is not None else (not smoke):
        saved_dir = str(save_student(model, spec, out_dir))

    completed = datetime.now(timezone.utc)
    record = {
        "experiment_id": exp_id,
        "spec": json.loads(spec.model_dump_json()),
        "device_id": spec.target_device_id,
        "status": "smoke_ok" if smoke else "built",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "align_steps": align_steps, "distill_steps": distill_steps_eff,
        "align_losses": align_losses,
        "distill_losses": distill_losses,
        "generations": gens,
        "student_dir": saved_dir,
        "run_config": run_meta,
        "note": ("smoke — assemble+align+distill+generate wired end-to-end."
                 if smoke else "built; run runners/eval_student.py for same-path scores."),
    }
    (out_dir / "build_record.json").write_text(json.dumps(record, indent=2))
    print(f"  ✅ {record['status']} → {out_dir/'build_record.json'}")
    return record


def main():
    ap = argparse.ArgumentParser(description="Construct a VLM student from a StudentSpec (ADR-0012)")
    ap.add_argument("--spec", required=True, help="Path to a StudentSpec JSON")
    ap.add_argument("--out", default=None, help="Output dir (default: artifacts/students/<hash12>)")
    ap.add_argument("--smoke", action="store_true", help="Tiny end-to-end smoke (proves the wiring)")
    ap.add_argument("--align-steps", type=int, default=None, help="Override align steps")
    ap.add_argument("--distill-steps", type=int, default=None, help="Override distill steps")
    ap.add_argument("--max-samples", type=int, default=None, help="Cap distill/align rows")
    args = ap.parse_args()

    spec = StudentSpec.model_validate_json(Path(args.spec).read_text())
    out = Path(args.out) if args.out else PROJECT_ROOT / "artifacts/students" / f"build_{spec.content_hash()[:12]}"
    build(spec, out, smoke=args.smoke, align_steps=args.align_steps,
          distill_steps=args.distill_steps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
