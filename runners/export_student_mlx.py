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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


# ── Step 2: vision + projector (scaffold) ────────────────────────────────────

def export_vision_projector(student_dir: Path, out_dir: Path) -> None:
    """Convert SigLIP to MLX and load the fresh MLP projector (projector.pt).

    TODO (ADR-0014 step 2): mlx-vlm provides SigLIP building blocks; the projector
    is a 2-layer MLP (Linear→GELU→Linear, hidden per spec). Load projector.pt
    state_dict and map into MLX arrays. Implemented after the LM half is verified.
    """
    raise NotImplementedError("step 2: SigLIP→MLX + projector port — see ADR-0014")


# ── Step 3: assembled parity gate (scaffold) ─────────────────────────────────

def parity_check(student_dir: Path, out_dir: Path) -> None:
    """Run the same image+prompt through PyTorch StudentVLM and the MLX student;
    assert matching greedy outputs, then re-score a small floor-adjusted slice.

    TODO (ADR-0014 step 3 — the go/no-go gate): mirrors the PyTorch forward
    (SigLIP encode → MLP project → prepend image embeds → Qwen decode). Compare
    decoded tokens on a handful of eval images; gate Swift work on a match.
    """
    raise NotImplementedError("step 3: assembled-forward parity — see ADR-0014")


def main():
    ap = argparse.ArgumentParser(description="Export a constructed student to MLX (ADR-0014)")
    ap.add_argument("--student", required=True, help="build dir, e.g. artifacts/students/build_d3423bc0155b")
    ap.add_argument("--step", default="lm", choices=["lm", "vision", "parity", "all"])
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


if __name__ == "__main__":
    main()
