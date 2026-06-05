"""
curate_eval_set.py
──────────────────
Curate the Stage A evaluation set from COCO val2017.

Selects 95 photos from COCO val2017 (the 5 existing stage_a_proxy images
round out to 100) with enforced diversity across scene/content categories.

Outputs:
  datasets/stage_a/photos/<coco_id>.jpg   — 95 selected images (copied)
  datasets/stage_a/captions.json          — COCO captions for 50 caption photos
  datasets/stage_a/vqa_template.json      — scaffold for 50 VQA pairs (manual fill-in)
  datasets/stage_a/selection_log.json     — audit log of selection decisions

Usage:
  python tools/curate_eval_set.py \
      --coco-images  datasets/coco_cache/val2017 \
      --coco-captions datasets/coco_cache/annotations/captions_val2017.json \
      --coco-instances datasets/coco_cache/annotations/instances_val2017.json \
      --out          datasets/stage_a \
      [--seed 42]
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ── Category buckets ──────────────────────────────────────────────────────────
# COCO supercategory names used to classify images.
# Images are assigned to the bucket of their most-frequent supercategory.

BUCKET_TARGETS = {
    "indoor_scene":   20,   # kitchen, living room, office, shop
    "outdoor_scene":  20,   # street, park, sports venue, nature
    "person_activity":20,   # activities, groups, sports action
    "animal":         15,   # pets, wildlife
    "vehicle":        10,   # cars, bikes, boats
    "food":           10,   # meals, produce
}
# Total = 95; remaining 5 slots come from existing stage_a_proxy img1-5.jpg
TOTAL_COCO = sum(BUCKET_TARGETS.values())   # 95

# COCO supercategories → our buckets
SUPERCATEGORY_TO_BUCKET = {
    "furniture":    "indoor_scene",
    "appliance":    "indoor_scene",
    "indoor":       "indoor_scene",
    "electronic":   "indoor_scene",
    "kitchen":      "indoor_scene",
    "food":         "food",
    "person":       "person_activity",
    "sports":       "outdoor_scene",
    "outdoor":      "outdoor_scene",
    "vehicle":      "vehicle",
    "animal":       "animal",
}

# Min caption length (chars) to accept a COCO caption as a reference
MIN_CAPTION_LEN = 40


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def image_bucket(image_id: int, id_to_supercats: dict[int, list[str]]) -> str:
    """Return the dominant bucket for an image, or 'other'."""
    supercats = id_to_supercats.get(image_id, [])
    counts = defaultdict(int)
    for sc in supercats:
        bucket = SUPERCATEGORY_TO_BUCKET.get(sc)
        if bucket:
            counts[bucket] += 1
    if not counts:
        return "other"
    return max(counts, key=counts.__getitem__)


def best_caption(captions: list[str]) -> str:
    """Pick the longest caption that meets the minimum length."""
    candidates = [c for c in captions if len(c) >= MIN_CAPTION_LEN]
    if candidates:
        return max(candidates, key=len)
    return max(captions, key=len) if captions else ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Curate Stage A evaluation set from COCO val2017")
    ap.add_argument("--coco-images",     required=True, help="Path to val2017/ directory")
    ap.add_argument("--coco-captions",   required=True, help="captions_val2017.json")
    ap.add_argument("--coco-instances",  required=True, help="instances_val2017.json")
    ap.add_argument("--out",             required=True, help="Output directory (datasets/stage_a)")
    ap.add_argument("--seed",            type=int, default=42, help="Random seed")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    coco_images_dir = Path(args.coco_images)
    out_dir         = Path(args.out)
    photos_dir      = out_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    print("Loading COCO annotations…")
    captions_data   = load_json(Path(args.coco_captions))
    instances_data  = load_json(Path(args.coco_instances))

    # Build image_id → filename map
    id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in captions_data["images"]
    }

    # Build image_id → list[supercategory] from instances
    cat_id_to_supercat: dict[int, str] = {
        cat["id"]: cat["supercategory"] for cat in instances_data["categories"]
    }
    id_to_supercats: dict[int, list[str]] = defaultdict(list)
    for ann in instances_data["annotations"]:
        sc = cat_id_to_supercat.get(ann["category_id"])
        if sc:
            id_to_supercats[ann["image_id"]].append(sc)

    # Build image_id → list[caption]
    id_to_captions: dict[int, list[str]] = defaultdict(list)
    for ann in captions_data["annotations"]:
        id_to_captions[ann["image_id"]].append(ann["caption"])

    # Only consider images that exist locally AND have captions
    available_ids = [
        img_id for img_id, fname in id_to_filename.items()
        if (coco_images_dir / fname).exists() and id_to_captions[img_id]
    ]
    print(f"  Available images with captions: {len(available_ids)}")

    # Bucket all available images
    bucket_to_ids: dict[str, list[int]] = defaultdict(list)
    for img_id in available_ids:
        bucket = image_bucket(img_id, id_to_supercats)
        bucket_to_ids[bucket].append(img_id)

    print("  Bucket sizes:")
    for b, ids in sorted(bucket_to_ids.items()):
        print(f"    {b:<20} {len(ids):>5}")

    # Select images per bucket
    selected: list[dict] = []
    selection_log: list[dict] = []

    for bucket, target in BUCKET_TARGETS.items():
        pool = bucket_to_ids.get(bucket, [])
        if len(pool) < target:
            print(f"  WARNING: {bucket} has only {len(pool)} images, need {target}")
        chosen = rng.sample(pool, min(target, len(pool)))
        for img_id in chosen:
            selected.append({"image_id": img_id, "bucket": bucket,
                              "filename": id_to_filename[img_id]})
            selection_log.append({"image_id": img_id, "bucket": bucket,
                                   "reason": f"sampled from bucket '{bucket}'"})

    print(f"\nSelected {len(selected)} COCO images.")

    # Copy images to out/photos/
    print("Copying images…")
    for item in selected:
        src = coco_images_dir / item["filename"]
        # Rename to zero-padded COCO ID: 000000123456.jpg
        dst = photos_dir / item["filename"]
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"  Copied to {photos_dir}")

    # ── Caption set (first 50 selected images) ──────────────────────────────
    caption_items = selected[:50]
    captions_out = {}
    for item in caption_items:
        caps = id_to_captions[item["image_id"]]
        captions_out[item["filename"]] = {
            "photo_id":  item["image_id"],
            "filename":  item["filename"],
            "bucket":    item["bucket"],
            "caption":   best_caption(caps),
            "all_captions": caps,
        }

    captions_path = out_dir / "captions.json"
    captions_path.write_text(json.dumps(captions_out, indent=2))
    print(f"  Captions written → {captions_path}  ({len(captions_out)} entries)")

    # ── VQA scaffold (next 50 selected images) ──────────────────────────────
    vqa_items = selected[50:100]
    # Question type rotation
    qtypes = (
        ["counting"] * 10 +
        ["activity"] * 15 +
        ["object_presence"] * 10 +
        ["color_attribute"] * 10 +
        ["scene_location"] * 5
    )
    vqa_out = []
    for i, item in enumerate(vqa_items):
        qtype = qtypes[i] if i < len(qtypes) else "activity"
        vqa_out.append({
            "id":            f"vqa_{i+1:03d}",
            "photo_id":      item["image_id"],
            "filename":      item["filename"],
            "bucket":        item["bucket"],
            "question_type": qtype,
            "question":      "TODO",   # fill in manually
            "answer":        "TODO",   # fill in manually
            "notes":         "",
        })

    vqa_path = out_dir / "vqa_template.json"
    vqa_path.write_text(json.dumps(vqa_out, indent=2))
    print(f"  VQA template written → {vqa_path}  ({len(vqa_out)} entries, fill in TODO fields)")

    # ── Selection log ────────────────────────────────────────────────────────
    log_path = out_dir / "selection_log.json"
    log_path.write_text(json.dumps({
        "seed": args.seed,
        "total_selected_coco": len(selected),
        "bucket_targets": BUCKET_TARGETS,
        "selections": selection_log,
    }, indent=2))
    print(f"  Selection log → {log_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("Stage A eval set scaffold complete.")
    print(f"  Photos:       {len(selected)} COCO + 5 existing = 100 total")
    print(f"  Caption set:  50 images with COCO reference captions")
    print(f"  VQA template: 50 images — fill in question/answer fields")
    print(f"  Output dir:   {out_dir}")
    print("\nNext steps:")
    print("  1. Fill in 'TODO' fields in vqa_template.json")
    print("     (rename to vqa.json when done)")
    print("  2. Run tools/hash_eval_set.py to generate manifest.json")
    print("═" * 55)


if __name__ == "__main__":
    main()
