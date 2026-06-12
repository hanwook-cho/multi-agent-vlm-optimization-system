"""
services/distillation_pipeline.py
─────────────────────────────────
Phase 2 Strategy B — caption-cache distillation.

Step 1 (this module): the **teacher** (Qwen2.5-VL-3B fp16) generates a caption
for every training image, **once**, and the result is cached to JSONL. Student
fine-tuning (runners/finetune_vlm.py) then trains against this cache without ever
re-running the teacher — that is what makes distillation affordable.

Design notes
------------
- **Reuses** the Qwen2.5-VL loader/infer from runners/generate_descriptions.py
  (same fp16 MPS path validated in P2-1.1) — no duplicate model code.
- **Resumable**: the cache is appended incrementally and already-captioned images
  are skipped, so an interrupted overnight run picks up where it left off.
- **Compute-gated**: generating the full cache is an explicit-approval step
  (Phase 2 plan). Start with a pilot (--limit) to validate the loop end-to-end.

Cache record (one JSON object per line):
    {"image": "<filename>", "caption": "<teacher caption>", "teacher": "...",
     "prompt": "...", "ts": "<iso8601>"}

Usage
-----
    # pilot (validate the loop on a small set)
    python services/distillation_pipeline.py \
        --images datasets/stage_a/photos --limit 50 \
        --out datasets/caption_cache/qwen25_3b_pilot.jsonl

    # full run (compute-gated — overnight)
    python services/distillation_pipeline.py \
        --images datasets/coco_train2017 --limit 50000 \
        --out datasets/caption_cache/qwen25_3b_coco50k.jsonl
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

# Reuse the validated Qwen2.5-VL fp16 loader/infer (P2-1.1).
from runners.generate_descriptions import load_qwen25vl, infer_qwen25vl, list_images

TEACHER_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# A descriptive caption prompt — the distillation target is a good open-ended
# description (the teacher's strength is grounding/reasoning, see P2-1.3).
CAPTION_PROMPT = "Describe this image in detail, including the main objects, their attributes, and the scene."


def _load_done(out_path: Path) -> set[str]:
    """Return the set of image filenames already present in the cache (for resume)."""
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["image"])
            except Exception:
                continue
    return done


def generate_caption_cache(
    image_dir: Path,
    out_path: Path,
    limit: int | None = None,
    prompt: str = CAPTION_PROMPT,
    teacher_model_id: str = TEACHER_MODEL_ID,
    device: str | None = None,
) -> int:
    """Generate teacher captions for images in `image_dir`, append to `out_path` (JSONL).

    Returns the number of NEW captions written this run. Resumable: images already
    in the cache are skipped.
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list_images(image_dir, limit)
    done = _load_done(out_path)
    todo = [n for n in names if n not in done]

    print(f"  teacher : {teacher_model_id}")
    print(f"  images  : {len(names)} total, {len(done)} cached, {len(todo)} to do")
    if not todo:
        print("  nothing to do — cache already complete for this set.")
        return 0

    print(f"  loading teacher on {device} …")
    model, processor = load_qwen25vl(device)

    written = 0
    # Append mode + flush per record → resumable and crash-safe.
    with out_path.open("a") as f:
        for i, name in enumerate(todo):
            img_path = image_dir / name
            try:
                image = Image.open(img_path).convert("RGB")
                caption = infer_qwen25vl(model, processor, image, prompt, device)
            except Exception as exc:
                print(f"\n  WARN {name}: {exc}")
                continue
            rec = {
                "image": name,
                "caption": caption,
                "teacher": teacher_model_id,
                "prompt": prompt,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            written += 1
            if i % 25 == 0:
                print(f"    [{i}/{len(todo)}] {name}: \"{caption[:60]}…\"")

    print(f"  done — wrote {written} new captions → {out_path}")
    return written


def main():
    ap = argparse.ArgumentParser(description="Generate teacher caption cache (Phase 2 Strategy B)")
    ap.add_argument("--images", required=True, help="Directory of training images")
    ap.add_argument("--out", required=True, help="Output JSONL cache path")
    ap.add_argument("--limit", type=int, default=None, help="Max images (use for the pilot, e.g. 50)")
    ap.add_argument("--prompt", default=CAPTION_PROMPT, help="Caption prompt for the teacher")
    ap.add_argument("--device", default=None, help="mps/cpu (auto-detected)")
    args = ap.parse_args()

    n = generate_caption_cache(
        image_dir=Path(args.images),
        out_path=Path(args.out),
        limit=args.limit,
        prompt=args.prompt,
        device=args.device,
    )
    print(f"\n✅ {n} new captions cached.")


if __name__ == "__main__":
    main()
