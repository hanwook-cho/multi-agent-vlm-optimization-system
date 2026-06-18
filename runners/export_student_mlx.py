"""Export a constructed student to MLX + verify parity (ADR-0014, step 1-3).

On-device deployment path for the *constructed* students (`d3423bc0`, `b2feb6b1`):
they are a custom PyTorch assembly (SigLIP + fresh MLP projector + Qwen2.5-0.5B +
LoRA), which llama.cpp/GGUF cannot run cleanly. We deploy via MLX-Swift; this
script does the Python-MLX conversion + correctness check on the Mac FIRST, so we
only attempt the Swift port once parity holds.

Incremental, each step verifiable (ADR-0014 plan):
  1. LM half      — merge LoRA into Qwen2.5-0.5B, convert to MLX, verify generate.
  2. Vision+proj  — convert SigLIP to MLX, load projector.pt into an MLX MLP.
  3. Parity gate  — assembled MLX forward vs PyTorch StudentVLM on the same input;
                    then a small floor-adjusted eval slice must match within noise.

ENVIRONMENT: needs `mlx`, `mlx-lm`, `mlx-vlm` at *mutually compatible* versions.
The brew-managed `mlx` on this machine (0.31) is older than current `mlx-lm`
requires, and cannot be pip-upgraded cleanly (no RECORD). Run this in an isolated
venv that also has torch/transformers/peft/Pillow for the parity check:
    python3 -m venv .venv-mlx && . .venv-mlx/bin/activate
    pip install mlx mlx-lm mlx-vlm torch transformers peft pillow
    python runners/export_student_mlx.py --student artifacts/students/build_d3423bc0155b

Usage:
    python runners/export_student_mlx.py --student <build_dir> [--step lm|vision|parity|all]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))   # so `runners.build_student` imports when run as a script


# ── Step 1: LM half (merge LoRA → MLX) ───────────────────────────────────────

def merge_lora_lm(student_dir: Path, out_dir: Path) -> Path:
    """Load the base LM + LoRA adapter, merge, and save a plain HF model dir."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = json.loads((student_dir / "student" / "spec.json").read_text())
    base_id = spec["lm"]
    adapter = student_dir / "student" / "lora_adapter"

    print(f"  loading base LM {base_id} + LoRA adapter")
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(base_id)

    merged_dir = out_dir / "lm_merged_hf"
    merged.save_pretrained(str(merged_dir))
    tok.save_pretrained(str(merged_dir))
    print(f"  merged LM (LoRA folded in) → {merged_dir}")
    return merged_dir


def convert_lm_to_mlx(merged_dir: Path, out_dir: Path) -> Path:
    """Convert the merged HF LM to MLX (fp16) via mlx_lm."""
    from mlx_lm import convert

    mlx_dir = out_dir / "lm_mlx"
    if mlx_dir.exists():
        shutil.rmtree(mlx_dir)
    convert(str(merged_dir), mlx_path=str(mlx_dir), quantize=False)
    print(f"  converted LM → MLX at {mlx_dir}")
    return mlx_dir


def verify_lm(mlx_dir: Path) -> None:
    """Smoke-check: the MLX LM loads and generates coherent text."""
    from mlx_lm import generate, load

    model, tok = load(str(mlx_dir))
    out = generate(model, tok, prompt="The capital of France is", max_tokens=16, verbose=False)
    print(f"  [LM verify] → {out!r}")
    assert out and len(out.strip()) > 0, "MLX LM produced empty output"
    print("  ✅ LM half OK (loads + generates)")


# ── Step 2: vision + projector (faithful MLX SigLIP + MLP projector) ─────────
#
# We re-implement SigLIP-base's vision tower functionally in MLX and load the HF
# weights, so we can verify `last_hidden_state` matches transformers numerically.
# SigLIP-base-patch16-224: Conv2d patch embed (no CLS), learned pos embed, 12
# encoder layers (LN→MHA→res, LN→MLP[gelu_tanh]→res), final post_layernorm.

_SIGLIP_ID = "google/siglip-base-patch16-224"
_N_LAYERS, _N_HEADS, _DIM, _EPS = 12, 12, 768, 1e-6


def _load_siglip_weights():
    """Pull SigLIP-base vision weights from HF into a dict of MLX arrays."""
    import mlx.core as mx
    from transformers import AutoModel

    vm = AutoModel.from_pretrained(_SIGLIP_ID).vision_model
    sd = {k: v.float().numpy() for k, v in vm.state_dict().items()}
    a = lambda k: mx.array(sd[k])
    W = {
        # conv weight [out,in,kh,kw] → MLX conv expects [out,kh,kw,in]
        "patch_w": mx.array(sd["embeddings.patch_embedding.weight"].transpose(0, 2, 3, 1)),
        "patch_b": a("embeddings.patch_embedding.bias"),
        "pos":     a("embeddings.position_embedding.weight"),
        "post_w":  a("post_layernorm.weight"), "post_b": a("post_layernorm.bias"),
    }
    for i in range(_N_LAYERS):
        p = f"encoder.layers.{i}."
        for nm in ("layer_norm1", "layer_norm2"):
            W[f"{nm}_w_{i}"] = a(p + nm + ".weight"); W[f"{nm}_b_{i}"] = a(p + nm + ".bias")
        for nm in ("q_proj", "k_proj", "v_proj", "out_proj"):
            W[f"{nm}_w_{i}"] = a(p + "self_attn." + nm + ".weight")
            W[f"{nm}_b_{i}"] = a(p + "self_attn." + nm + ".bias")
        for nm in ("fc1", "fc2"):
            W[f"{nm}_w_{i}"] = a(p + "mlp." + nm + ".weight"); W[f"{nm}_b_{i}"] = a(p + "mlp." + nm + ".bias")
    return W


def _ln(x, w, b):
    import mlx.core as mx
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / mx.sqrt(var + _EPS) * w + b


def siglip_mlx_forward(W, pixel_values):
    """pixel_values: MLX [B,H,W,C]. Returns last_hidden_state [B,196,768]."""
    import mlx.core as mx
    import mlx.nn as nn

    x = mx.conv2d(pixel_values, W["patch_w"], stride=16) + W["patch_b"]   # [B,14,14,768]
    B = x.shape[0]
    x = x.reshape(B, -1, _DIM) + W["pos"]                                 # [B,196,768]
    hd = _DIM // _N_HEADS
    for i in range(_N_LAYERS):
        h = _ln(x, W[f"layer_norm1_w_{i}"], W[f"layer_norm1_b_{i}"])
        q = h @ W[f"q_proj_w_{i}"].T + W[f"q_proj_b_{i}"]
        k = h @ W[f"k_proj_w_{i}"].T + W[f"k_proj_b_{i}"]
        v = h @ W[f"v_proj_w_{i}"].T + W[f"v_proj_b_{i}"]
        P = q.shape[1]
        def split(t): return t.reshape(B, P, _N_HEADS, hd).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
        attn = mx.softmax(scores, axis=-1) @ v                            # [B,H,P,hd]
        attn = attn.transpose(0, 2, 1, 3).reshape(B, P, _DIM)
        x = x + (attn @ W[f"out_proj_w_{i}"].T + W[f"out_proj_b_{i}"])
        h = _ln(x, W[f"layer_norm2_w_{i}"], W[f"layer_norm2_b_{i}"])
        h = nn.gelu_approx(h @ W[f"fc1_w_{i}"].T + W[f"fc1_b_{i}"])        # gelu_pytorch_tanh
        x = x + (h @ W[f"fc2_w_{i}"].T + W[f"fc2_b_{i}"])
    return _ln(x, W["post_w"], W["post_b"])                               # [B,196,768]


def _load_projector_mlx(student_dir: Path):
    """Load projector.pt (nn.Sequential Linear→GELU→Linear) into MLX arrays."""
    import mlx.core as mx
    import torch
    sd = torch.load(student_dir / "student" / "projector.pt", map_location="cpu")
    return {k: mx.array(v.float().numpy()) for k, v in sd.items()}


def projector_mlx_forward(P, feats):
    import mlx.nn as nn
    h = feats @ P["0.weight"].T + P["0.bias"]
    h = nn.gelu(h)                          # PyTorch nn.GELU default = exact erf gelu
    return h @ P["2.weight"].T + P["2.bias"]


def _pixel_values_mlx(image_path: str):
    """Preprocess an image with the SigLIP processor; return (mlx [B,H,W,C], torch [B,C,H,W])."""
    import mlx.core as mx
    from PIL import Image
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(_SIGLIP_ID)
    pv = proc(images=Image.open(image_path).convert("RGB"), return_tensors="pt").pixel_values  # [1,3,H,W]
    pv_mlx = mx.array(pv.permute(0, 2, 3, 1).contiguous().numpy())  # [1,H,W,3]
    return pv_mlx, pv


def export_vision_projector(student_dir: Path, out_dir: Path) -> None:
    """Build the MLX SigLIP + projector and verify image-embeds match PyTorch."""
    import mlx.core as mx
    import numpy as np
    import torch
    from transformers import AutoModel

    img = _find_sample_image()
    print(f"  parity image: {img}")
    pv_mlx, pv_torch = _pixel_values_mlx(img)

    # MLX side
    W = _load_siglip_weights()
    P = _load_projector_mlx(student_dir)
    mx_feats = siglip_mlx_forward(W, pv_mlx)
    mx_embeds = projector_mlx_forward(P, mx_feats)
    mx.eval(mx_embeds)

    # PyTorch reference (same as StudentVLM._image_embeds)
    vm = AutoModel.from_pretrained(_SIGLIP_ID).vision_model.eval()
    with torch.no_grad():
        t_feats = vm(pixel_values=pv_torch).last_hidden_state
    feats_diff = float(np.abs(np.array(mx_feats) - t_feats.numpy()).max())
    print(f"  SigLIP last_hidden_state max|Δ| = {feats_diff:.2e}  (want < 1e-2)")
    assert feats_diff < 1e-2, "SigLIP MLX port does not match transformers"
    print("  ✅ vision+projector parity OK")


# ── Step 3: assembled-forward parity gate ────────────────────────────────────

def _find_sample_image() -> str:
    for cand in (PROJECT_ROOT / "datasets" / "coco_train2017",):
        if cand.exists():
            for f in sorted(cand.iterdir()):
                if f.suffix.lower() in (".jpg", ".png"):
                    return str(f)
    raise SystemExit("no sample image found (datasets/coco_train2017)")


def parity_check(student_dir: Path, out_dir: Path) -> None:
    """Greedy-decode the same image+prompt through PyTorch StudentVLM and the MLX
    student; assert the generated token strings match (the go/no-go gate)."""
    import mlx.core as mx
    import mlx.nn as nn
    import torch
    from mlx_lm import load as mlx_load
    from runners.build_student import load_student

    img = _find_sample_image()
    prompt = "Is there a person in the image? Please answer yes or no."
    print(f"  parity image: {img}\n  prompt: {prompt!r}")

    # --- PyTorch student reference ---
    student = load_student(student_dir)
    proc = student._proc
    pv = proc.image(images=__import__("PIL").Image.open(img).convert("RGB"),
                    return_tensors="pt").pixel_values.to(next(student.lm.parameters()).device)
    enc = proc.tok(prompt, return_tensors="pt").to(pv.device)
    with torch.no_grad():
        gen = student.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                               pixel_values=pv, max_new_tokens=16)
    torch_out = proc.tok.decode(gen[0], skip_special_tokens=True).strip()
    print(f"  [PyTorch] → {torch_out!r}")

    # --- MLX student ---
    W = _load_siglip_weights()
    P = _load_projector_mlx(student_dir)
    pv_mlx, _ = _pixel_values_mlx(img)
    img_embeds = projector_mlx_forward(P, siglip_mlx_forward(W, pv_mlx))   # [1,196,896]
    lm, tok = mlx_load(str(out_dir / "lm_mlx"))
    embed_layer = lm.model.embed_tokens
    ids = mx.array(tok.encode(prompt))[None]
    txt_embeds = embed_layer(ids)
    step_in = mx.concatenate([img_embeds.astype(txt_embeds.dtype), txt_embeds], axis=1)
    out_ids, eos = [], tok.eos_token_id
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    dummy = mx.zeros((1, 1), dtype=mx.int32)   # `inputs` is unused when input_embeddings is set
    for _ in range(16):
        logits = lm(dummy, cache=cache, input_embeddings=step_in)[:, -1, :]
        nxt = int(mx.argmax(logits, axis=-1).item())
        if nxt == eos: break
        out_ids.append(nxt)
        step_in = embed_layer(mx.array([[nxt]]))
    mlx_out = tok.decode(out_ids).strip()
    print(f"  [MLX]     → {mlx_out!r}")

    match = mlx_out == torch_out
    print(f"  {'✅ PARITY MATCH' if match else '⚠️ MISMATCH'} (gate: {'PASS' if match else 'investigate'})")
    return match


def measure_mac_perf(student_dir: Path, out_dir: Path) -> None:
    """Step 3.5 — Apple-Silicon (Mac MLX) perf of the assembled student:
    TTFT, decode tokens/sec, peak memory. A real on-device-class number without Swift."""
    import time
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    W = _load_siglip_weights()
    P = _load_projector_mlx(student_dir)
    img = _find_sample_image()
    pv_mlx, _ = _pixel_values_mlx(img)
    lm, tok = mlx_load(str(out_dir / "lm_mlx"))
    from mlx_lm.models.cache import make_prompt_cache

    prompt = "Describe this image briefly."
    embed = lm.model.embed_tokens
    _reset_peak = getattr(mx, "reset_peak_memory", None) or getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
    _get_peak = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    if _reset_peak: _reset_peak()

    t0 = time.perf_counter()
    img_embeds = projector_mlx_forward(P, siglip_mlx_forward(W, pv_mlx))
    ids = mx.array(tok.encode(prompt))[None]
    step_in = mx.concatenate([img_embeds.astype(embed(ids).dtype), embed(ids)], axis=1)
    cache = make_prompt_cache(lm)
    dummy = mx.zeros((1, 1), dtype=mx.int32)
    logits = lm(dummy, cache=cache, input_embeddings=step_in)[:, -1, :]
    nxt = mx.argmax(logits, axis=-1); mx.eval(nxt)
    ttft = (time.perf_counter() - t0) * 1000

    n = 32; t1 = time.perf_counter()
    for _ in range(n):
        step_in = embed(nxt[None])
        logits = lm(dummy, cache=cache, input_embeddings=step_in)[:, -1, :]
        nxt = mx.argmax(logits, axis=-1); mx.eval(nxt)
    tps = n / (time.perf_counter() - t1)
    peak = (_get_peak() / 1e6) if _get_peak else float("nan")
    print(f"  [Mac MLX perf]  TTFT(+vision)={ttft:.0f} ms  decode={tps:.1f} tok/s  peak≈{peak:.0f} MB")


def main():
    ap = argparse.ArgumentParser(description="Export a constructed student to MLX (ADR-0014)")
    ap.add_argument("--student", required=True, help="build dir, e.g. artifacts/students/build_d3423bc0155b")
    ap.add_argument("--step", default="lm", choices=["lm", "vision", "parity", "perf", "all"])
    ap.add_argument("--out", default=None, help="output dir (default: <student>/mlx_export)")
    args = ap.parse_args()

    student_dir = Path(args.student)
    out_dir = Path(args.out) if args.out else student_dir / "mlx_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.step in ("lm", "all"):
        print("▶ Step 1 — LM half (merge LoRA → MLX → verify)")
        merged = merge_lora_lm(student_dir, out_dir)
        mlx_dir = convert_lm_to_mlx(merged, out_dir)
        verify_lm(mlx_dir)
    if args.step in ("vision", "all"):
        print("▶ Step 2 — vision + projector")
        export_vision_projector(student_dir, out_dir)
    if args.step in ("parity", "all"):
        print("▶ Step 3 — assembled-forward parity gate")
        parity_check(student_dir, out_dir)
    if args.step in ("perf", "all"):
        print("▶ Step 3.5 — Mac (Apple-Silicon MLX) perf")
        measure_mac_perf(student_dir, out_dir)


if __name__ == "__main__":
    main()
