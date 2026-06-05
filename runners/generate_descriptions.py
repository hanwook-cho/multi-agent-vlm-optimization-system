"""
generate_descriptions.py
────────────────────────
Generate open-ended image descriptions from each Phase 0 VLM on Mac.
Outputs a predictions JSON per model, ready for compute_clip_score.py.

Usage:
    python generate_descriptions.py \
        --images datasets/stage_a_proxy/photos \
        --out    artifacts/clip_preds/ \
        --models LFM2-VL-450M SmolVLM-500M MiniCPM-V-4.6 FastVLM-0.5B

    # Or score all models:
    python generate_descriptions.py \
        --images datasets/stage_a_proxy/photos \
        --out    artifacts/clip_preds/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

PROMPT = "Describe what you see in this image."

# ── Model loaders (reuse patterns from eval_vlmeval.py) ──────────────────────

def load_lfm2(device):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model_id = "LiquidAI/LFM2-VL-450M"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    return model, processor

def infer_lfm2(model, processor, image: Image.Image, prompt: str, device) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    decoded = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def load_smolvlm(device):
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device).eval()
    return model, processor

def infer_smolvlm(model, processor, image: Image.Image, prompt: str, device) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    decoded = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def load_minicpm(device):
    from transformers import MiniCPMV4_6ForConditionalGeneration, AutoProcessor
    from huggingface_hub import snapshot_download
    import glob
    model_id = "openbmb/MiniCPM-V-4.6"
    # Find the local snapshot that has preprocessor_config.json
    cache_base = os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4.6/snapshots"
    )
    snapshots = glob.glob(os.path.join(cache_base, "*/preprocessor_config.json"))
    if snapshots:
        local_path = os.path.dirname(snapshots[0])
    else:
        local_path = snapshot_download(model_id, local_files_only=True)
    processor = AutoProcessor.from_pretrained(local_path)
    model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        local_path, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device).eval()
    return model, processor

def infer_minicpm(model, processor, image: Image.Image, prompt: str, device) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=text, return_tensors="pt").to(device)
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    decoded = processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
    return decoded


def load_fastvlm(device):
    """FastVLM uses LlavaForConditionalGeneration with MobileCLIP encoder."""
    from transformers import AutoProcessor
    from transformers import LlavaForConditionalGeneration
    model_id = "apple/FastVLM-0.5B"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    return model, processor

def infer_fastvlm(model, processor, image: Image.Image, prompt: str, device) -> str:
    from transformers import CLIPImageProcessor
    # FastVLM needs pixel_values from CLIPImageProcessor, not LlavaProcessor
    img_processor = CLIPImageProcessor.from_pretrained(
        "apple/FastVLM-0.5B", subfolder="image_processor", trust_remote_code=True
    )
    pixel_values = img_processor(images=image, return_tensors="pt").pixel_values.to(device)
    # Manual tokenisation
    full_prompt = f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = processor.tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(inputs=input_ids, pixel_values=pixel_values,
                              max_new_tokens=128, do_sample=False)
    decoded = processor.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return decoded.strip()


# ── Registry ─────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "LFM2-VL-450M":  (load_lfm2,     infer_lfm2),
    "SmolVLM-500M":  (load_smolvlm,   infer_smolvlm),
    "MiniCPM-V-4.6": (load_minicpm,   infer_minicpm),
    "FastVLM-0.5B":  (load_fastvlm,   infer_fastvlm),
}

SAMPLE_IMAGES = [f"img{i}.jpg" for i in range(1, 6)]


# ── Main ─────────────────────────────────────────────────────────────────────

def run_model(model_key: str, image_dir: Path, out_dir: Path, device: str):
    load_fn, infer_fn = MODEL_REGISTRY[model_key]
    out_file = out_dir / f"{model_key}_preds.json"

    if out_file.exists():
        print(f"  {model_key}: already exists, skipping → {out_file}")
        return

    print(f"\n{'═'*55}")
    print(f"  Loading {model_key}…")
    model, processor = load_fn(device)

    predictions = []
    for img_name in SAMPLE_IMAGES:
        img_path = image_dir / img_name
        if not img_path.exists():
            print(f"  WARNING: {img_path} not found, skipping")
            continue
        image = Image.open(img_path).convert("RGB")
        print(f"  {img_name}… ", end="", flush=True)
        text = infer_fn(model, processor, image, PROMPT, device)
        print(f'"{text[:80]}"')
        predictions.append({"image": img_name, "text": text})

    result = {"model_key": model_key, "prompt": PROMPT, "predictions": predictions}
    out_file.write_text(json.dumps(result, indent=2))
    print(f"  saved → {out_file}")

    # Free memory
    del model, processor
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="Generate image descriptions for CLIP-score evaluation")
    ap.add_argument("--images",  default="datasets/stage_a_proxy/photos",
                    help="Directory containing sample images")
    ap.add_argument("--out",     default="artifacts/clip_preds",
                    help="Output directory for predictions JSONs")
    ap.add_argument("--models",  nargs="+", default=list(MODEL_REGISTRY),
                    help=f"Models to run (default: all). Choices: {list(MODEL_REGISTRY)}")
    ap.add_argument("--device",  default=None,
                    help="Device (mps/cuda/cpu). Auto-detected if not specified.")
    args = ap.parse_args()

    device = args.device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    image_dir = Path(args.images)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown = [m for m in args.models if m not in MODEL_REGISTRY]
    if unknown:
        print(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY)}")
        sys.exit(1)

    for model_key in args.models:
        run_model(model_key, image_dir, out_dir, device)

    print(f"\n✅ Done. Predictions in {out_dir}/")
    print("Next: python runners/compute_clip_score.py"
          f" --images {args.images} --preds {out_dir}/ --out artifacts/clip_scores/")


if __name__ == "__main__":
    main()
