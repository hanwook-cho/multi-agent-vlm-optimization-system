"""
tools/fetch_coco_subset.py
──────────────────────────
Fetch a deterministic subset of COCO train2017 images for Phase 2 distillation.

Reads the (already-present) captions_train2017.json manifest, takes a seeded random
sample of N image file names, and downloads them from the COCO image server. Only
images are fetched — the teacher (Qwen2.5-VL-3B) generates the captions, so COCO's
own captions are not used.

Resumable (skips files already on disk), parallel, deterministic for a given seed.

Usage:
    python tools/fetch_coco_subset.py --n 5000 --out datasets/coco_train2017
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST = PROJECT_ROOT / "datasets/coco_cache/annotations/captions_train2017.json"
IMG_BASE = "http://images.cocodataset.org/train2017"


def select_file_names(n: int, seed: int) -> list[str]:
    data = json.loads(MANIFEST.read_text())
    names = sorted({img["file_name"] for img in data["images"]})  # deterministic base order
    rng = random.Random(seed)
    return rng.sample(names, min(n, len(names)))


def _fetch_one(name: str, out_dir: Path) -> tuple[str, bool]:
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return name, True
    try:
        urlretrieve(f"{IMG_BASE}/{name}", dest)
        return name, True
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return name, False


def main():
    ap = argparse.ArgumentParser(description="Fetch a COCO train2017 image subset")
    ap.add_argument("--n", type=int, default=5000, help="Number of images")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed (reproducible)")
    ap.add_argument("--out", default="datasets/coco_train2017", help="Output image dir")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    names = select_file_names(args.n, args.seed)
    print(f"Fetching {len(names)} COCO train2017 images (seed={args.seed}) → {out_dir}")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_fetch_one, n, out_dir): n for n in names}
        for i, fut in enumerate(as_completed(futs)):
            _, success = fut.result()
            ok += success
            fail += (not success)
            if i % 250 == 0:
                print(f"  [{i}/{len(names)}]  ok={ok} fail={fail}", flush=True)

    print(f"Done: {ok} downloaded/present, {fail} failed → {out_dir}")


if __name__ == "__main__":
    main()
