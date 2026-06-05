"""
compute_clip_score.py
─────────────────────
Compute CLIPScore for (image, description) pairs produced by VLMs.

CLIPScore (Hessel et al. 2021) measures semantic alignment between a
generated description and its source image using CLIP embeddings.
Score = 100 × max(0, cos_sim(CLIP_img(I), CLIP_txt(T)))
Range: 0–100. Higher = better alignment. ~25–35 is typical for good captions.

Usage
─────
# Score a single model from a predictions file:
python compute_clip_score.py \
    --images  datasets/stage_a_proxy/photos \
    --preds   artifacts/clip_preds/FastVLM-0.5B_preds.json \
    --out     artifacts/clip_scores/FastVLM-0.5B_clip.json

# Score all models in a predictions directory:
python compute_clip_score.py \
    --images  datasets/stage_a_proxy/photos \
    --preds   artifacts/clip_preds/ \
    --out     artifacts/clip_scores/

Predictions JSON format:
    {
      "model_key": "FastVLM-0.5B",
      "predictions": [
        {"image": "sample1.jpg", "text": "Two cats sleeping on a red blanket."},
        ...
      ]
    }
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── Config ────────────────────────────────────────────────────────────────────

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"   # large for best quality
SCALE = 100.0                                       # multiply cosine → 0-100 range


# ── Helpers ───────────────────────────────────────────────────────────────────

class CLIPScorer:
    """Loads CLIP once and exposes score(images, texts) → list[float]."""

    def __init__(self, model_id: str = CLIP_MODEL_ID, device: str | None = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        print(f"Loading CLIP ({model_id}) on {device}…")
        self.model     = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        print("CLIP ready.")

    @torch.no_grad()
    def score(self, images: list[Image.Image], texts: list[str]) -> list[float]:
        """
        Returns per-pair CLIPScore in [0, 100].
        images and texts must have the same length.
        """
        assert len(images) == len(texts), "images and texts must be same length"

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        outputs   = self.model(**inputs)
        img_emb   = outputs.image_embeds   # (N, D)
        txt_emb   = outputs.text_embeds    # (N, D)

        # Normalise
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        # Per-pair cosine similarity (diagonal of the full matrix)
        cosine = (img_emb * txt_emb).sum(dim=-1)   # (N,)
        scores = (SCALE * cosine.clamp(min=0)).tolist()
        return scores


def _load_predictions(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_image(image_dir: Path, filename: str) -> Image.Image:
    # Try with and without extension
    for name in [filename, filename + ".jpg", filename + ".png"]:
        p = image_dir / name
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"Image not found: {filename} in {image_dir}")


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


# ── Main scoring logic ────────────────────────────────────────────────────────

def score_predictions(
    preds_path: Path,
    image_dir: Path,
    scorer: CLIPScorer,
) -> dict:
    """Score one predictions file. Returns result dict."""
    data    = _load_predictions(preds_path)
    model   = data.get("model_key", preds_path.stem)
    preds   = data["predictions"]

    images  = [_load_image(image_dir, p["image"]) for p in preds]
    texts   = [p["text"] for p in preds]
    scores  = scorer.score(images, texts)

    per_image = [
        {"image": p["image"], "text": p["text"], "clip_score": round(s, 4)}
        for p, s in zip(preds, scores)
    ]
    mean_score = sum(scores) / len(scores)
    std_score  = _stddev(scores)

    return {
        "model_key":       model,
        "clip_model":      CLIP_MODEL_ID,
        "mean_clip_score": round(mean_score, 4),
        "std_clip_score":  round(std_score, 4),
        "n":               len(scores),
        "per_image":       per_image,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compute CLIPScore for VLM predictions")
    ap.add_argument("--images", required=True,
                    help="Directory containing the source images")
    ap.add_argument("--preds",  required=True,
                    help="Predictions JSON file or directory of JSON files")
    ap.add_argument("--out",    required=True,
                    help="Output JSON file or directory")
    ap.add_argument("--model",  default=CLIP_MODEL_ID,
                    help=f"CLIP model ID (default: {CLIP_MODEL_ID})")
    args = ap.parse_args()

    image_dir  = Path(args.images)
    preds_path = Path(args.preds)
    out_path   = Path(args.out)

    scorer = CLIPScorer(model_id=args.model)

    if preds_path.is_dir():
        # Score all JSON files in the directory
        preds_files = sorted(preds_path.glob("*.json"))
        out_path.mkdir(parents=True, exist_ok=True)
        all_results = []
        for pf in preds_files:
            print(f"\n── {pf.name} ──")
            result = score_predictions(pf, image_dir, scorer)
            all_results.append(result)
            out_file = out_path / (pf.stem + "_clip.json")
            out_file.write_text(json.dumps(result, indent=2))
            print(f"  mean CLIPScore: {result['mean_clip_score']:.2f} ± {result['std_clip_score']:.2f}")
            print(f"  saved → {out_file}")

        # Summary table
        print("\n" + "═" * 55)
        print(f"{'Model':<30} {'CLIP':>8} {'±σ':>6}")
        print("─" * 55)
        for r in sorted(all_results, key=lambda x: -x["mean_clip_score"]):
            print(f"{r['model_key']:<30} {r['mean_clip_score']:>8.2f} {r['std_clip_score']:>6.2f}")
        print("═" * 55)

    else:
        # Score a single file
        print(f"\n── {preds_path.name} ──")
        result = score_predictions(preds_path, image_dir, scorer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  mean CLIPScore: {result['mean_clip_score']:.2f} ± {result['std_clip_score']:.2f}")
        for p in result["per_image"]:
            print(f"  {p['image']}: {p['clip_score']:.2f}  \"{p['text'][:60]}...\"")
        print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
