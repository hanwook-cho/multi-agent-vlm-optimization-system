"""Build a ScienceQA MCQ distill cache — MMBench-distribution training data (P2-B1).

The construction student floors on MMBench because it was trained on COCO (POPE's
distribution), which is off-distribution for MMBench's science/knowledge/reasoning
questions (see docs/observations/2026-06-16-p2b1-rehearsal-full-epoch.md and the
follow-on). ScienceQA is the single closest public match to MMBench's dominant
distribution AND is natively multiple-choice with gold answers — so we train on
ScienceQA gold (distribution-matched, correct labels) rather than teacher-distilled
COCO MCQ.

Emits rows in the unified cache format consumed by build_student._load_rows:
    {"image": "<file>", "prompt": "<q>\\n<opts><MCQ suffix>", "target": "<A-D>", ...}

Filters to image-bearing questions with 2-4 choices (our eval format is A-D).
Usage:
    python runners/build_scienceqa_cache.py --limit 2500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Must match runners.eval_vlmeval.MCQ_PROMPT_SUFFIX / distillation_pipeline.MCQ_TRAIN_SUFFIX.
MCQ_TRAIN_SUFFIX = " Answer with only the letter A, B, C, or D."
LETTERS = "ABCD"


def format_record(question: str, hint: str, choices: list, answer) -> tuple[str, str] | None:
    """Pure: (question, hint, choices, answer-index) → (prompt, target-letter) or None.

    Returns None for rows we don't keep: not 2-4 choices, or an out-of-range answer
    (our eval format is A-D). Kept separate from I/O so the contract is unit-testable.
    """
    if not (2 <= len(choices) <= 4):
        return None
    if answer is None or not (0 <= answer < len(choices)):
        return None
    opt_block = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(choices))
    q = f"{hint.strip()}\n{question.strip()}" if (hint or "").strip() else (question or "").strip()
    return f"{q}\n{opt_block}{MCQ_TRAIN_SUFFIX}", LETTERS[answer]


def build(limit: int, split: str, out_path: Path, img_dir: Path) -> int:
    from datasets import load_dataset

    img_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream so we don't materialise the whole split; ScienceQA mixes image and
    # text-only rows, and we only want image-bearing multiple-choice questions.
    ds = load_dataset("derek-thomas/ScienceQA", split=split, streaming=True)

    written = skipped_noimg = skipped_choices = 0
    with out_path.open("w") as fh:
        for i, ex in enumerate(ds):
            if written >= limit:
                break
            img = ex.get("image")
            if img is None:
                skipped_noimg += 1
                continue
            fmt = format_record(ex.get("question") or "", ex.get("hint") or "",
                                ex.get("choices") or [], ex.get("answer"))
            if fmt is None:
                skipped_choices += 1
                continue
            prompt, target = fmt

            fname = f"scienceqa_{split}_{i:06d}.png"
            try:
                img.convert("RGB").save(img_dir / fname)
            except Exception:
                continue

            rec = {
                "image": fname,
                "prompt": prompt,
                "target": target,
                "kind": "mcq",
                "source": "scienceqa",
                "subject": ex.get("subject"),
            }
            fh.write(json.dumps(rec) + "\n")
            written += 1
            if written % 250 == 0:
                print(f"  …{written} written")

    print(f"✅ {written} rows → {out_path}  (images → {img_dir})")
    print(f"   skipped: {skipped_noimg} text-only, {skipped_choices} bad-choice-count")
    return written


def main():
    ap = argparse.ArgumentParser(description="Build a ScienceQA MCQ distill cache")
    ap.add_argument("--limit", type=int, default=2500, help="Max image-MCQ rows to write")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="datasets/caption_cache/scienceqa_mcq.jsonl")
    ap.add_argument("--img-dir", default="datasets/scienceqa_images")
    args = ap.parse_args()
    build(args.limit, args.split, Path(args.out), Path(args.img_dir))


if __name__ == "__main__":
    main()
