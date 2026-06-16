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

# Task-aligned (QA) mode — P2-D2. Caption-only distillation regressed the student
# on the MCQ benchmarks (P2-D1) because it taught captioning, not the measured
# question-answering skill. In "qa" mode the teacher generates grounded
# question→answer pairs (the skill POPE/RealWorldQA/MMBench actually test), so the
# student distills the right behavior. Each image yields multiple training records.
QA_GEN_PROMPT = (
    "Generate 3 diverse question-answer pairs that test understanding of this image. "
    "Include at least one yes/no question about whether a specific object is present, "
    "and one about a visual attribute or spatial relationship. Keep answers short and "
    "factual. Format strictly, one per line:\n"
    "Q: <question>\nA: <answer>"
)

# Multiple-choice mode — the student is at the MMBench/RealWorldQA floor because it
# never trained on the MCQ format (question + options A-D → answer letter). The
# teacher generates grounded multiple-choice items so the student learns to pick a
# letter. The training prompt mirrors the eval (build_mcq_question + MCQ suffix) so
# train/eval format matches.
MCQ_GEN_PROMPT = (
    "Create a multiple-choice question that tests understanding of THIS image. "
    "Give exactly four options labeled A, B, C, D, with exactly one correct answer. "
    "Make the wrong options plausible. Format strictly:\n"
    "Question: <question>\nA. <option>\nB. <option>\nC. <option>\nD. <option>\n"
    "Answer: <letter A-D>"
)
# Must match runners.eval_vlmeval.MCQ_PROMPT_SUFFIX so train/eval prompts align.
MCQ_TRAIN_SUFFIX = " Answer with only the letter A, B, C, or D."


def _parse_qa(text: str) -> list[tuple[str, str]]:
    """Parse 'Q: ... / A: ...' pairs from the teacher's output."""
    pairs: list[tuple[str, str]] = []
    q = None
    for line in text.splitlines():
        line = line.strip()
        if line[:2].upper() == "Q:":
            q = line[2:].strip()
        elif line[:2].upper() == "A:" and q:
            a = line[2:].strip()
            if q and a:
                pairs.append((q, a))
            q = None
    return pairs


def _parse_mcq(text: str) -> dict | None:
    """Parse a 'Question/A./B./C./D./Answer:' block → {prompt, target} or None.

    prompt = the question + the four options + the MCQ suffix (matching the eval
    format); target = the single correct letter. Returns None if malformed.
    """
    import re
    question = None
    options: dict[str, str] = {}
    answer = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("question:"):
            question = line.split(":", 1)[1].strip()
        elif re.match(r"^[A-D][.)]\s", line):
            options[line[0].upper()] = line[2:].strip()
        elif line.lower().startswith("answer:"):
            m = re.search(r"[A-D]", line.split(":", 1)[1].upper())
            if m:
                answer = m.group(0)
    if not question or len(options) != 4 or answer not in options:
        return None
    opt_block = "\n".join(f"{L}. {options[L]}" for L in ("A", "B", "C", "D"))
    return {"prompt": f"{question}\n{opt_block}{MCQ_TRAIN_SUFFIX}", "target": answer}


# ── B1.1: balanced hard-negative QA (the P2-D2 fix) ──────────────────────────
# P2-D2 regressed because the teacher Q&A asked mostly about objects that ARE
# present → the student learned an always-"Yes" presence prior (POPE acc 50,
# recall 100). The fix is to teach grounded *absence* too: emit equal numbers of
# present-object ("Yes") and confirmed-absent-object ("No") presence questions,
# the exact balance POPE measures. Labels stay grounded in the teacher (it lists
# what it sees and confirms what it doesn't), not in random sampling.

# COCO-80 thing classes — the candidate pool for plausible hard negatives.
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

VISIBLE_PROBE = (
    "List every distinct object you can clearly see in this image as lowercase "
    "singular nouns, comma-separated. Only list what is actually visible."
)
ABSENCE_PROBE_TMPL = (
    "For EACH object in this list, say whether it is clearly visible in the image. "
    "Answer with exactly one line per object in the form '<object>: yes' or "
    "'<object>: no'.\nObjects: {cands}"
)


def _is_yesno(answer: str) -> bool:
    return answer.strip().lower().rstrip(".") in {"yes", "no"}


def _parse_object_list(text: str) -> list[str]:
    """Parse a comma/newline-separated object list into normalized lowercase nouns."""
    raw = text.replace("\n", ",").split(",")
    objs: list[str] = []
    for tok in raw:
        t = tok.strip().lower().strip(".-•* ")
        # drop leading articles / numbering noise
        for pre in ("a ", "an ", "the "):
            if t.startswith(pre):
                t = t[len(pre):]
        if t and t.replace(" ", "").isalpha() and len(t) <= 20:
            objs.append(t)
    return objs


def _parse_presence_labels(text: str) -> dict[str, bool]:
    """Parse '<object>: yes/no' lines → {object: present?}."""
    labels: dict[str, bool] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        obj, _, val = line.partition(":")
        v = val.strip().lower().rstrip(".")
        obj = obj.strip().lower().strip("-•* ")
        if obj and v in {"yes", "no"}:
            labels[obj] = (v == "yes")
    return labels


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


def generate_qa_cache(
    image_dir: Path,
    out_path: Path,
    limit: int | None = None,
    teacher_model_id: str = TEACHER_MODEL_ID,
    device: str | None = None,
) -> int:
    """Task-aligned (P2-D2): teacher generates grounded Q&A pairs per image.

    Writes one record per (image, question, answer) — multiple per image — as
    {image, prompt: <question>, target: <answer>, kind: "qa", ...}. The fine-tune
    runner trains the student to answer the question (the measured MCQ skill),
    unlike caption-only distillation (P2-D1) which taught captioning and regressed.

    Resumable at image granularity (skips images already in the cache).
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list_images(image_dir, limit)
    done = _load_done(out_path)
    todo = [n for n in names if n not in done]

    print(f"  teacher : {teacher_model_id}  (mode=qa)")
    print(f"  images  : {len(names)} total, {len(done)} cached, {len(todo)} to do")
    if not todo:
        print("  nothing to do — cache already complete for this set.")
        return 0

    print(f"  loading teacher on {device} …")
    model, processor = load_qwen25vl(device)

    written = pairs_total = 0
    with out_path.open("a") as f:
        for i, name in enumerate(todo):
            try:
                image = Image.open(image_dir / name).convert("RGB")
                raw = infer_qwen25vl(model, processor, image, QA_GEN_PROMPT, device)
                pairs = _parse_qa(raw)
            except Exception as exc:
                print(f"\n  WARN {name}: {exc}")
                continue
            for q, a in pairs:
                f.write(json.dumps({
                    "image": name, "prompt": q, "target": a, "kind": "qa",
                    "teacher": teacher_model_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                pairs_total += 1
            f.flush()
            written += 1
            if i % 25 == 0:
                ex = pairs[0] if pairs else ("", "")
                print(f"    [{i}/{len(todo)}] {name}: {len(pairs)} Q&A  e.g. Q:\"{ex[0][:40]}\" A:\"{ex[1][:30]}\"")

    print(f"  done — {pairs_total} Q&A pairs from {written} images → {out_path}")
    return pairs_total


def generate_balanced_qa_cache(
    image_dir: Path,
    out_path: Path,
    limit: int | None = None,
    pairs_per_image: int = 2,
    seed: int = 0,
    teacher_model_id: str = TEACHER_MODEL_ID,
    device: str | None = None,
) -> int:
    """B1.1: task-aligned Q&A with BALANCED grounded presence questions (the P2-D2 fix).

    Per image (3 teacher calls):
      1. QA_GEN  → keep the OPEN (non yes/no) pairs for attribute/spatial skill.
      2. VISIBLE → grounded list of present objects (positives → "Yes").
      3. ABSENCE → confirm a sample of COCO-80 objects NOT in the visible list are
                   absent (grounded negatives → "No").
    Emits min(#present, #absent, pairs_per_image) of EACH presence class so the
    yes/no balance is ~50/50 — preventing the always-"Yes" collapse that broke P2-D2.

    Records carry a `kind`: "open" | "presence". Resumable at image granularity.
    Compute note: 3 teacher calls/image (~3× the plain qa cache) — pilot with --limit.
    """
    import random
    rng = random.Random(seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list_images(image_dir, limit)
    done = _load_done(out_path)
    todo = [n for n in names if n not in done]

    print(f"  teacher : {teacher_model_id}  (mode=qa_balanced)")
    print(f"  images  : {len(names)} total, {len(done)} cached, {len(todo)} to do")
    if not todo:
        print("  nothing to do — cache already complete for this set.")
        return 0

    print(f"  loading teacher on {device} …")
    model, processor = load_qwen25vl(device)

    written = pairs_total = n_yes = n_no = n_open = 0
    with out_path.open("a") as f:
        for i, name in enumerate(todo):
            try:
                image = Image.open(image_dir / name).convert("RGB")
                # 1. open QA (attribute/spatial — keep only non-yes/no)
                open_pairs = [(q, a) for q, a in _parse_qa(
                    infer_qwen25vl(model, processor, image, QA_GEN_PROMPT, device))
                    if not _is_yesno(a)]
                # 2. present objects (grounded positives)
                present = _parse_object_list(
                    infer_qwen25vl(model, processor, image, VISIBLE_PROBE, device))
                present = [o for o in dict.fromkeys(present) if o in COCO80]
                # 3. confirm grounded negatives from non-present COCO-80
                cand_pool = [o for o in COCO80 if o not in present]
                cands = rng.sample(cand_pool, min(8, len(cand_pool)))
                labels = _parse_presence_labels(infer_qwen25vl(
                    model, processor, image,
                    ABSENCE_PROBE_TMPL.format(cands=", ".join(cands)), device))
                absent = [o for o in cands if labels.get(o) is False]
            except Exception as exc:
                print(f"\n  WARN {name}: {exc}")
                continue

            rng.shuffle(present); rng.shuffle(absent)
            m = min(len(present), len(absent), pairs_per_image)  # 50/50 presence
            recs = []
            for q, a in open_pairs:
                recs.append({"prompt": q, "target": a, "kind": "open"})
            for o in present[:m]:
                recs.append({"prompt": f"Is there a {o} in the image?", "target": "Yes",
                             "kind": "presence"})
            for o in absent[:m]:
                recs.append({"prompt": f"Is there a {o} in the image?", "target": "No",
                             "kind": "presence"})

            ts = datetime.now(timezone.utc).isoformat()
            for r in recs:
                f.write(json.dumps({"image": name, **r, "teacher": teacher_model_id,
                                    "ts": ts}) + "\n")
                pairs_total += 1
                n_yes += r["target"] == "Yes" and r["kind"] == "presence"
                n_no += r["target"] == "No" and r["kind"] == "presence"
                n_open += r["kind"] == "open"
            f.flush()
            written += 1
            if i % 10 == 0:
                print(f"    [{i}/{len(todo)}] {name}: +{len(recs)} ({m}Y/{m}N presence, "
                      f"{len(open_pairs)} open)")

    print(f"  done — {pairs_total} pairs from {written} images "
          f"(presence {n_yes}Y/{n_no}N, open {n_open}) → {out_path}")
    return pairs_total


def generate_mcq_cache(
    image_dir: Path,
    out_path: Path,
    limit: int | None = None,
    teacher_model_id: str = TEACHER_MODEL_ID,
    device: str | None = None,
) -> int:
    """Teacher generates grounded multiple-choice items (question + A-D → letter).

    Writes {image, prompt, target: <letter>, kind: "mcq", ...}. The prompt mirrors
    the eval format (options + 'Answer with only the letter...'), so the student
    learns the MCQ format the MMBench/RealWorldQA benchmarks measure. Resumable.
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list_images(image_dir, limit)
    done = _load_done(out_path)
    todo = [n for n in names if n not in done]

    print(f"  teacher : {teacher_model_id}  (mode=mcq)")
    print(f"  images  : {len(names)} total, {len(done)} cached, {len(todo)} to do")
    if not todo:
        print("  nothing to do — cache already complete for this set.")
        return 0
    print(f"  loading teacher on {device} …")
    model, processor = load_qwen25vl(device)

    written = 0
    with out_path.open("a") as f:
        for i, name in enumerate(todo):
            try:
                image = Image.open(image_dir / name).convert("RGB")
                raw = infer_qwen25vl(model, processor, image, MCQ_GEN_PROMPT, device)
                rec = _parse_mcq(raw)
            except Exception as exc:
                print(f"\n  WARN {name}: {exc}")
                continue
            if rec is None:
                continue
            f.write(json.dumps({"image": name, **rec, "kind": "mcq",
                                "teacher": teacher_model_id,
                                "ts": datetime.now(timezone.utc).isoformat()}) + "\n")
            f.flush()
            written += 1
            if i % 10 == 0:
                print(f"    [{i}/{len(todo)}] {name}: {rec['target']}  {rec['prompt'][:50]!r}")

    print(f"  done — {written} MCQ items from {len(todo)} images → {out_path}")
    return written


def main():
    ap = argparse.ArgumentParser(description="Generate teacher distillation cache (Phase 2 Strategy B)")
    ap.add_argument("--images", required=True, help="Directory of training images")
    ap.add_argument("--out", required=True, help="Output JSONL cache path")
    ap.add_argument("--mode", choices=["caption", "qa", "qa_balanced", "mcq"], default="caption",
                    help="caption (P2-D1), qa = task-aligned Q&A (P2-D2), "
                         "qa_balanced = grounded 50/50 yes/no presence + open (B1.1), "
                         "mcq = multiple-choice items (off the MMBench floor)")
    ap.add_argument("--limit", type=int, default=None, help="Max images (use for the pilot, e.g. 50)")
    ap.add_argument("--pairs-per-image", type=int, default=2,
                    help="qa_balanced: max present/absent presence Qs per class per image")
    ap.add_argument("--prompt", default=CAPTION_PROMPT, help="Caption prompt (caption mode only)")
    ap.add_argument("--seed", type=int, default=0, help="qa_balanced negative-sampling seed")
    ap.add_argument("--device", default=None, help="mps/cpu (auto-detected)")
    args = ap.parse_args()

    if args.mode == "mcq":
        n = generate_mcq_cache(
            image_dir=Path(args.images), out_path=Path(args.out),
            limit=args.limit, device=args.device,
        )
        print(f"\n✅ {n} MCQ items cached (multiple-choice, off the MMBench floor).")
    elif args.mode == "qa_balanced":
        n = generate_balanced_qa_cache(
            image_dir=Path(args.images), out_path=Path(args.out),
            limit=args.limit, pairs_per_image=args.pairs_per_image,
            seed=args.seed, device=args.device,
        )
        print(f"\n✅ {n} pairs cached (task-aligned + balanced hard negatives, B1.1).")
    elif args.mode == "qa":
        n = generate_qa_cache(
            image_dir=Path(args.images), out_path=Path(args.out),
            limit=args.limit, device=args.device,
        )
        print(f"\n✅ {n} Q&A pairs cached (task-aligned, P2-D2).")
    else:
        n = generate_caption_cache(
            image_dir=Path(args.images), out_path=Path(args.out),
            limit=args.limit, prompt=args.prompt, device=args.device,
        )
        print(f"\n✅ {n} new captions cached.")


if __name__ == "__main__":
    main()
