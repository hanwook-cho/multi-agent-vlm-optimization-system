"""
runners/finetune_vlm.py
───────────────────────
Phase 2 Strategy B — fine-tune a student VLM on the teacher caption cache (LoRA).

Reads the JSONL cache produced by services/distillation_pipeline.py and LoRA-tunes
a small student (default LFM2-VL-450M, hypothesis H-P2-005) to imitate the
Qwen2.5-VL-3B teacher's captions. Saves the LoRA adapter and a MetricsReport.

Default LoRA config (Phase 2 plan):
    rank=16, alpha=32, target = q_proj / v_proj / o_proj
    3 epochs, lr 2e-4, cosine decay, batch 4 × grad-accum 8 (eff. 32), greedy.

NOTE
----
- Requires `peft` (`pip install peft`). Imported lazily so the scaffold loads
  without it; the trainer errors with a clear message if it's absent.
- Compute-gated: a real run is multi-hour on the M4. Validate with a canary first
  (small cache, 1 epoch) — Phase 2 task P2-3.5.
- VLM collation is model-specific. This targets LFM2-VL's AutoProcessor chat
  template; the canary run is what confirms label masking / shapes are right.

Usage
-----
    # canary (validate the loop)
    python runners/finetune_vlm.py \
        --cache datasets/caption_cache/qwen25_3b_pilot.jsonl \
        --images datasets/stage_a/photos \
        --out artifacts/students/lfm2vl_distill_canary \
        --epochs 1 --max-samples 100

    # full run (compute-gated)
    python runners/finetune_vlm.py \
        --cache datasets/caption_cache/qwen25_3b_coco50k.jsonl \
        --images datasets/coco_train2017 \
        --out artifacts/students/lfm2vl_distill_s0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_STUDENT = "LiquidAI/LFM2-VL-450M"
CAPTION_PROMPT = "Describe this image in detail, including the main objects, their attributes, and the scene."

# LoRA defaults (Phase 2 plan)
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q_proj", "v_proj", "o_proj"]


def _require_peft():
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "fine-tuning requires `peft` — install it first:\n"
            "    pip install peft\n"
            f"(import error: {exc})"
        )


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_cache(cache_path: Path, image_dir: Path, max_samples: int | None) -> list[dict]:
    """Load (image_path, caption) pairs from the JSONL cache; skip missing images."""
    rows: list[dict] = []
    for line in cache_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        img = image_dir / rec["image"]
        if not img.exists() or not rec.get("caption"):
            continue
        rows.append({"image_path": str(img), "caption": rec["caption"]})
        if max_samples and len(rows) >= max_samples:
            break
    return rows


class _CaptionCollator:
    """Collate (image, caption) → model inputs with labels masked on the prompt.

    Only the assistant caption tokens contribute to the loss; the user prompt and
    image-placeholder tokens are masked to -100.
    """

    def __init__(self, processor, prompt: str):
        self.processor = processor
        self.prompt = prompt

    def __call__(self, batch: list[dict]) -> dict:
        texts, images = [], []
        prompt_lens = []
        for ex in batch:
            image = Image.open(ex["image_path"]).convert("RGB")
            user_msg = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt},
            ]}]
            # Prompt-only render (for masking length), then full with the caption.
            prompt_text = self.processor.apply_chat_template(user_msg, add_generation_prompt=True)
            full_text = prompt_text + ex["caption"] + self.processor.tokenizer.eos_token
            texts.append(full_text)
            images.append(image)
            prompt_lens.append(len(self.processor.tokenizer(prompt_text).input_ids))

        # Group images per text (nested) so batched samples map 1:1 to their image;
        # a flat list fails for batch>1 ("number of images in text [1,1,..] and images [N]").
        enc = self.processor(text=texts, images=[[im] for im in images],
                             return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        # Mask the prompt span (everything before the caption) per example.
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        enc["labels"] = labels
        return enc


def finetune(
    cache_path: Path,
    image_dir: Path,
    out_dir: Path,
    base_model: str = DEFAULT_STUDENT,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 8,
    max_samples: int | None = None,
    seed: int = 0,
    prompt: str = CAPTION_PROMPT,
) -> Path:
    """LoRA-fine-tune the student on the caption cache. Returns the adapter dir."""
    _require_peft()
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              Trainer, TrainingArguments)

    torch.manual_seed(seed)
    device = _device()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_cache(cache_path, image_dir, max_samples)
    if not rows:
        raise SystemExit(f"no usable (image, caption) pairs in {cache_path}")
    print(f"  student   : {base_model}")
    print(f"  cache     : {len(rows)} pairs from {cache_path.name}")
    print(f"  LoRA      : r={LORA_R} alpha={LORA_ALPHA} targets={LORA_TARGETS}")
    print(f"  schedule  : {epochs} epochs, lr={lr}, eff-batch={batch_size*grad_accum}, seed={seed}")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)

    lora = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    from datasets import Dataset
    ds = Dataset.from_list(rows)
    collator = _CaptionCollator(processor, prompt)

    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        # MPS is auto-detected in transformers 5.x (use_mps_device was removed).
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)

    started = datetime.now(timezone.utc)
    trainer.train()
    completed = datetime.now(timezone.utc)

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    # Minimal run record (full MetricsReport is written after eval, P2-4/6).
    (out_dir / "train_meta.json").write_text(json.dumps({
        "base_model": base_model,
        "cache": str(cache_path),
        "n_pairs": len(rows),
        "epochs": epochs, "lr": lr, "seed": seed,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "targets": LORA_TARGETS},
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "adapter_dir": str(adapter_dir),
    }, indent=2))
    print(f"  ✅ adapter saved → {adapter_dir}")
    return adapter_dir


def main():
    ap = argparse.ArgumentParser(description="LoRA-distill a student VLM from the teacher caption cache")
    ap.add_argument("--cache", required=True, help="Teacher caption cache JSONL")
    ap.add_argument("--images", required=True, help="Directory with the cached images")
    ap.add_argument("--out", required=True, help="Output dir for the adapter + meta")
    ap.add_argument("--base-model", default=DEFAULT_STUDENT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=None, help="Cap pairs (canary)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    finetune(
        cache_path=Path(args.cache), image_dir=Path(args.images), out_dir=Path(args.out),
        base_model=args.base_model, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_samples=args.max_samples, seed=args.seed,
    )


if __name__ == "__main__":
    main()
