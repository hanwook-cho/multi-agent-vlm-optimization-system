"""
smoke_test_models.py
────────────────────
Pre-flight check: load each Phase 0 VLM and run one forward pass.

Designed to catch transformers compatibility regressions (e.g. after a
pip upgrade) before wasting a full measurement run. Runs in < 5 minutes
total on Mac M-series with all four models.

Usage:
    # Test all models
    python tools/smoke_test_models.py

    # Test a subset
    python tools/smoke_test_models.py --models LFM2-VL-450M SmolVLM-500M

    # Custom image + dry-run (just check imports, no inference)
    python tools/smoke_test_models.py --image path/to/test.jpg --dry-run

    # Show what would run without loading anything
    python tools/smoke_test_models.py --list

Exit codes:
    0 — all tested models passed
    1 — one or more models failed
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "datasets" / "stage_a_proxy" / "photos" / "img1.jpg"
FALLBACK_IMAGE = PROJECT_ROOT / "datasets" / "stage_a" / "photos" / "000000391895.jpg"

SMOKE_PROMPT   = "What is in this image? Answer in one word."
MAX_NEW_TOKENS = 16   # fast — just enough to confirm decode works


# ── Model loaders (same patterns as runners/generate_descriptions.py) ─────────

def load_lfm2(device):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model_id = "LiquidAI/LFM2-VL-450M"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    return model, processor

def infer_lfm2(model, processor, image, device):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": SMOKE_PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def load_smolvlm(device):
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device).eval()
    return model, processor

def infer_smolvlm(model, processor, image, device):
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": SMOKE_PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def load_minicpm(device):
    import glob
    from transformers import MiniCPMV4_6ForConditionalGeneration, AutoProcessor
    from huggingface_hub import snapshot_download
    model_id = "openbmb/MiniCPM-V-4.6"
    cache_base = os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4.6/snapshots"
    )
    snapshots = glob.glob(os.path.join(cache_base, "*/preprocessor_config.json"))
    local_path = os.path.dirname(snapshots[0]) if snapshots else \
                 snapshot_download(model_id, local_files_only=True)
    processor = AutoProcessor.from_pretrained(local_path)
    model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        local_path, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()
    return model, processor

def infer_minicpm(model, processor, image, device):
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": SMOKE_PROMPT},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=text, return_tensors="pt").to(device)
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


def load_fastvlm(device):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model_id = "apple/FastVLM-0.5B"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    return model, processor

def infer_fastvlm(model, processor, image, device):
    from transformers import CLIPImageProcessor
    img_processor = CLIPImageProcessor.from_pretrained(
        "apple/FastVLM-0.5B", subfolder="image_processor", trust_remote_code=True
    )
    pixel_values = img_processor(images=image, return_tensors="pt").pixel_values.to(device)
    full_prompt = f"<|im_start|>user\n<image>\n{SMOKE_PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = processor.tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(inputs=input_ids, pixel_values=pixel_values,
                             max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.tokenizer.decode(out[0][input_ids.shape[1]:],
                                      skip_special_tokens=True).strip()


# ── Registry ─────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "LFM2-VL-450M":  (load_lfm2,     infer_lfm2),
    "SmolVLM-500M":  (load_smolvlm,   infer_smolvlm),
    "MiniCPM-V-4.6": (load_minicpm,   infer_minicpm),
    "FastVLM-0.5B":  (load_fastvlm,   infer_fastvlm),
}


# ── Test runner ───────────────────────────────────────────────────────────────

def _free(model, processor, device):
    del model, processor
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def smoke_test_model(model_key: str, image: Image.Image, device: str,
                     dry_run: bool = False) -> dict:
    """
    Returns a result dict:
        status:    "PASS" | "FAIL" | "DRY_RUN"
        load_s:    load time in seconds (or None)
        infer_s:   inference time in seconds (or None)
        output:    model output string (or None)
        error:     exception string (or None)
    """
    load_fn, infer_fn = MODEL_REGISTRY[model_key]

    if dry_run:
        return {"status": "DRY_RUN", "load_s": None, "infer_s": None,
                "output": None, "error": None}

    model = processor = None
    try:
        t0 = time.perf_counter()
        model, processor = load_fn(device)
        load_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        output = infer_fn(model, processor, image, device)
        infer_s = time.perf_counter() - t1

        _free(model, processor, device)

        if not output:
            return {"status": "FAIL", "load_s": load_s, "infer_s": infer_s,
                    "output": output, "error": "Empty output"}
        return {"status": "PASS", "load_s": load_s, "infer_s": infer_s,
                "output": output, "error": None}

    except Exception as e:
        if model is not None:
            try:
                _free(model, processor, device)
            except Exception:
                pass
        return {"status": "FAIL", "load_s": None, "infer_s": None,
                "output": None, "error": traceback.format_exc().strip()}


# ── Formatting ────────────────────────────────────────────────────────────────

def _status_icon(status):
    return {"PASS": "✅", "FAIL": "❌", "DRY_RUN": "⏭️ "}.get(status, "?")


def print_summary(results: dict):
    print()
    print("┌─────────────────────┬──────────┬────────────┬────────────┬───────────────────────────────────┐")
    print("│ Model               │ Status   │ Load (s)   │ Infer (s)  │ Output / Error                    │")
    print("├─────────────────────┼──────────┼────────────┼────────────┼───────────────────────────────────┤")
    for key, r in results.items():
        icon   = _status_icon(r["status"])
        load_s = f"{r['load_s']:.1f}" if r["load_s"] is not None else "—"
        infer_s = f"{r['infer_s']:.2f}" if r["infer_s"] is not None else "—"
        if r["status"] == "PASS":
            detail = (r["output"] or "")[:33]
        elif r["status"] == "DRY_RUN":
            detail = "(skipped)"
        else:
            first_line = (r["error"] or "").splitlines()[-1][:33]
            detail = first_line
        print(f"│ {key:<19} │ {icon} {r['status']:<6} │ {load_s:>10} │ {infer_s:>10} │ {detail:<33} │")
    print("└─────────────────────┴──────────┴────────────┴────────────┴───────────────────────────────────┘")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pre-flight smoke test for all Phase 0 VLMs")
    ap.add_argument("--models",  nargs="+", default=list(MODEL_REGISTRY),
                    help=f"Models to test. Choices: {list(MODEL_REGISTRY)}")
    ap.add_argument("--image",   default=None,
                    help="Path to test image. Auto-resolved if not specified.")
    ap.add_argument("--device",  default=None,
                    help="Device override (mps/cuda/cpu). Auto-detected if omitted.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Verify imports/paths only — skip model load and inference.")
    ap.add_argument("--list",    action="store_true",
                    help="Print available models and exit.")
    args = ap.parse_args()

    if args.list:
        print("Available models:")
        for k in MODEL_REGISTRY:
            print(f"  {k}")
        return

    # Validate model names
    unknown = [m for m in args.models if m not in MODEL_REGISTRY]
    if unknown:
        print(f"ERROR: Unknown models: {unknown}")
        print(f"Available: {list(MODEL_REGISTRY)}")
        sys.exit(1)

    # Resolve device
    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Resolve test image
    if args.image:
        image_path = Path(args.image)
    elif DEFAULT_IMAGE.exists():
        image_path = DEFAULT_IMAGE
    elif FALLBACK_IMAGE.exists():
        image_path = FALLBACK_IMAGE
    else:
        # Last resort: generate a synthetic 224×224 RGB image
        image_path = None

    if not args.dry_run:
        if image_path is not None:
            if not image_path.exists():
                print(f"ERROR: Image not found: {image_path}")
                sys.exit(1)
            image = Image.open(image_path).convert("RGB")
            print(f"Test image : {image_path}  ({image.size[0]}×{image.size[1]})")
        else:
            # Synthetic fallback: solid grey 224×224
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            print("Test image : synthetic 224×224 grey (no real image found)")
    else:
        image = None
        print("Dry-run mode: imports + paths only, no model load.")

    print(f"Device     : {device}")
    print(f"Models     : {', '.join(args.models)}")
    print(f"Max tokens : {MAX_NEW_TOKENS}")
    print()

    # Run smoke tests
    results = {}
    t_total = time.perf_counter()
    for model_key in args.models:
        print(f"  Testing {model_key}… ", end="", flush=True)
        r = smoke_test_model(model_key, image, device, dry_run=args.dry_run)
        results[model_key] = r
        icon = _status_icon(r["status"])
        if r["status"] == "PASS":
            print(f"{icon} PASS  (load={r['load_s']:.1f}s  infer={r['infer_s']:.2f}s)"
                  f"  → \"{(r['output'] or '')[:60]}\"")
        elif r["status"] == "DRY_RUN":
            print(f"{icon} DRY_RUN")
        else:
            short_err = (r["error"] or "").splitlines()[-1][:80]
            print(f"{icon} FAIL  → {short_err}")
            if r["error"]:
                # Print full traceback indented
                for line in r["error"].splitlines():
                    print(f"      {line}")

    total_s = time.perf_counter() - t_total

    print_summary(results)

    n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
    n_fail = sum(1 for r in results.values() if r["status"] == "FAIL")
    n_skip = sum(1 for r in results.values() if r["status"] == "DRY_RUN")
    print(f"\n  {n_pass} passed  {n_fail} failed  {n_skip} skipped  "
          f"({total_s:.1f}s total)\n")

    if n_fail > 0:
        print("  SMOKE TEST FAILED — fix the errors above before running experiments.")
        sys.exit(1)
    else:
        print("  All models healthy. Safe to proceed with experiments.")


if __name__ == "__main__":
    main()
