"""
build_vqa_from_coco.py
──────────────────────
Pull VQA v2 question/answer pairs from COCO annotations for the
Stage A eval set's 45 VQA-designated images.

COCO VQA v2 has ~214k questions for val2017 images (~43 per image on average).
We select one high-quality Q/A pair per image, favouring:
  1. Correct answers that are unambiguous (high agreement in VQA v2 majority vote)
  2. Question type matches the pre-assigned type bucket
  3. Short, exact-match-friendly answers ("yes", "no", a number, a colour, a noun)

Usage:
    python tools/build_vqa_from_coco.py \
        --vqa-questions  datasets/coco_cache/v2_OpenEnded_mscoco_val2017_questions.json \
        --vqa-anns       datasets/coco_cache/v2_mscoco_val2017_annotations.json \
        --template       datasets/stage_a/vqa_template.json \
        --out            datasets/stage_a/vqa.json
"""

import argparse
import json
import re
from pathlib import Path


# VQA v2 answer_type values: "yes/no", "number", "other"
# question_type strings (prefix of question): "is", "are", "how many", "what color", "what", ...

QTYPE_PREFERENCE = {
    "counting":        ["how many"],
    "activity":        ["what is the", "what are the", "what is he", "what is she",
                        "what are they", "what sport", "what is this person"],
    "object_presence": ["is there", "are there", "is this", "is the", "is a", "is an"],
    "color_attribute": ["what color", "what colour"],
    "scene_location":  ["where", "is this indoors", "is this outside", "is this inside",
                        "what room", "what type of"],
}

# Minimum number of annotators agreeing on the answer for it to be accepted
MIN_AGREEMENT = 3   # out of 10 annotators in VQA v2


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def normalise_answer(ans: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    ans = ans.lower().strip()
    ans = re.sub(r"[^\w\s]", "", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def best_answer(answers: list[dict]) -> tuple[str, int]:
    """
    VQA v2 stores 10 free-text answers per question.
    Return (most_common_answer, agreement_count).
    """
    from collections import Counter
    normed = [normalise_answer(a["answer"]) for a in answers]
    most_common, count = Counter(normed).most_common(1)[0]
    return most_common, count


def question_matches_type(question: str, qtype: str) -> bool:
    q = question.lower()
    for prefix in QTYPE_PREFERENCE.get(qtype, []):
        if q.startswith(prefix):
            return True
    return False


def select_qa_for_image(
    image_id: int,
    qtype: str,
    id_to_questions: dict,
    qid_to_answers: dict,
) -> dict | None:
    """
    Pick the best Q/A pair for this image and question type.
    Falls back to any high-agreement Q/A if no type match found.
    """
    questions = id_to_questions.get(image_id, [])
    if not questions:
        return None

    candidates = []
    for q in questions:
        qid = q["question_id"]
        ann = qid_to_answers.get(qid)
        if ann is None:
            continue
        answer, agreement = best_answer(ann["answers"])
        if agreement < MIN_AGREEMENT:
            continue
        type_match = question_matches_type(q["question"], qtype)
        candidates.append({
            "question_id":  qid,
            "question":     q["question"],
            "answer":       answer,
            "agreement":    agreement,
            "answer_type":  ann.get("answer_type", "other"),
            "type_match":   type_match,
        })

    if not candidates:
        return None

    # Sort: type-matched first, then by agreement descending
    candidates.sort(key=lambda c: (-int(c["type_match"]), -c["agreement"]))
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description="Build VQA pairs from COCO VQA v2 annotations")
    ap.add_argument("--vqa-questions", required=True)
    ap.add_argument("--vqa-anns",      required=True)
    ap.add_argument("--template",      required=True,
                    help="vqa_template.json produced by curate_eval_set.py")
    ap.add_argument("--out",           required=True, help="Output vqa.json path")
    args = ap.parse_args()

    print("Loading VQA v2 annotations…")
    questions_data = load_json(Path(args.vqa_questions))
    anns_data      = load_json(Path(args.vqa_anns))
    template       = json.loads(Path(args.template).read_text())

    # Build lookup tables
    id_to_questions: dict[int, list] = {}
    for q in questions_data["questions"]:
        id_to_questions.setdefault(q["image_id"], []).append(q)

    qid_to_answers: dict[int, dict] = {
        a["question_id"]: a for a in anns_data["annotations"]
    }

    print(f"  {len(questions_data['questions']):,} questions, "
          f"{len(anns_data['annotations']):,} answer sets loaded")
    print(f"  {len(template)} images in VQA template")

    # Fill in each template entry
    vqa_out = []
    missing = 0
    for entry in template:
        image_id = entry["photo_id"]
        qtype    = entry["question_type"]

        result = select_qa_for_image(image_id, qtype, id_to_questions, qid_to_answers)

        if result is None:
            print(f"  WARNING: no Q/A found for image {image_id} (type={qtype})")
            missing += 1
            vqa_out.append({**entry, "question": "MISSING", "answer": "MISSING"})
            continue

        vqa_out.append({
            "id":               entry["id"],
            "photo_id":         image_id,
            "filename":         entry["filename"],
            "bucket":           entry["bucket"],
            "question_type":    qtype,
            "question":         result["question"],
            "answer":           result["answer"],
            "vqa_question_id":  result["question_id"],
            "answer_agreement": result["agreement"],
            "answer_type":      result["answer_type"],
            "type_match":       result["type_match"],
        })

        match_str = "✓ type match" if result["type_match"] else "~ fallback"
        print(f"  {entry['id']} ({qtype:18}) [{match_str}] "
              f"agree={result['agreement']}/10  "
              f"Q: {result['question'][:55]}  A: {result['answer']}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(vqa_out, indent=2))

    type_match_count = sum(1 for v in vqa_out if v.get("type_match"))
    print(f"\n{'='*60}")
    print(f"VQA pairs written → {out_path}")
    print(f"  Total:       {len(vqa_out)}")
    print(f"  Type-matched: {type_match_count} / {len(vqa_out)}")
    print(f"  Missing:      {missing}")
    if missing:
        print(f"  ⚠️  {missing} image(s) had no VQA v2 entry — fill in manually")
    print(f"{'='*60}")
    print("\nNext: run tools/hash_eval_set.py to update manifest.json with final hash")


if __name__ == "__main__":
    main()
