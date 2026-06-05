"""
hash_eval_set.py
────────────────
Generate a frozen manifest for the Stage A evaluation set.

Computes per-file SHA-256 for every photo, captions.json, and vqa.json,
then computes a single manifest_hash (SHA-256 of the sorted manifest
contents). This hash is the eval set's identity — every ExperimentConfig
that uses this eval set stores it as `eval_set_hash`.

Usage:
    python tools/hash_eval_set.py --eval-dir datasets/stage_a

Writes:
    datasets/stage_a/manifest.json
"""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Hash-pin the Stage A eval set")
    ap.add_argument("--eval-dir", default="datasets/stage_a",
                    help="Eval set root directory")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    photos_dir = eval_dir / "photos"

    if not photos_dir.exists():
        print(f"ERROR: {photos_dir} does not exist. Run curate_eval_set.py first.")
        raise SystemExit(1)

    print(f"Hashing eval set at {eval_dir} …")

    # Hash all photos
    photo_files = sorted(photos_dir.glob("*.jpg")) + sorted(photos_dir.glob("*.png"))
    file_hashes: dict[str, str] = {}
    for p in photo_files:
        rel = str(p.relative_to(eval_dir))
        file_hashes[rel] = sha256_file(p)
        print(f"  {rel}  {file_hashes[rel][:12]}…")

    # Hash annotation files if present
    for name in ("captions.json", "vqa.json", "vqa_template.json"):
        p = eval_dir / name
        if p.exists():
            rel = name
            file_hashes[rel] = sha256_file(p)
            print(f"  {rel}  {file_hashes[rel][:12]}…")

    # Manifest hash = SHA-256 of deterministic JSON of sorted file_hashes
    canonical = json.dumps(dict(sorted(file_hashes.items())), separators=(",", ":"))
    manifest_hash = sha256_string(canonical)

    # Load counts
    captions_count = 0
    vqa_count = 0
    for name in ("captions.json",):
        p = eval_dir / name
        if p.exists():
            captions_count = len(json.loads(p.read_text()))
    for name in ("vqa.json", "vqa_template.json"):
        p = eval_dir / name
        if p.exists():
            vqa_count = len(json.loads(p.read_text()))
            break

    manifest = {
        "version": "1.0.0",
        "created": str(date.today()),
        "description": "Stage A evaluation set — Phase 0 reference, frozen",
        "sources": ["coco_val2017 (seed=42)", "stage_a_proxy img1-5.jpg"],
        "n_photos": len(photo_files),
        "n_caption_photos": captions_count,
        "n_vqa_pairs": vqa_count,
        "files": file_hashes,
        "manifest_hash": manifest_hash,
    }

    manifest_path = eval_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print("═" * 55)
    print(f"manifest.json written → {manifest_path}")
    print(f"  Photos:         {len(photo_files)}")
    print(f"  Caption photos: {captions_count}")
    print(f"  VQA pairs:      {vqa_count}")
    print(f"  manifest_hash:  {manifest_hash}")
    print("═" * 55)
    print()
    print("Record this hash in ExperimentConfig.eval_set_hash for all Phase 1 runs:")
    print(f"  {manifest_hash}")


if __name__ == "__main__":
    main()
