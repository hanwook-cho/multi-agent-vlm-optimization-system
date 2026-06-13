# Observation: Caption-Only Distillation Regresses the Student (Phase 2 Strategy B pilot)

**Date:** 2026-06-13
**Context:** Phase 2 Strategy B pilot — distill Qwen2.5-VL-3B → LFM2-VL-450M and measure MCQ gain
**Verdict:** **Negative — caption-only LoRA distillation degraded the student on every MCQ benchmark.** The naive approach is wrong; the pilot caught it before the full 50K / 3-seed run.

---

## Setup

- **Teacher:** Qwen2.5-VL-3B (fp16) captioned 5,000 COCO train2017 images (prompt: "Describe this image in detail…").
- **Student:** LFM2-VL-450M, LoRA (r=16, α=32, q/v/o_proj), 3 epochs, lr 2e-4, batch-1, seed 0. Final `train_loss` 3.575 (from canary 9.51 → it learned the captioning objective).
- **Eval:** POPE / RealWorldQA / MMBench, 100-sample slices, **same fp16 transformers path** for baseline and student (P2-1.3 methodology).

---

## Result

| Benchmark | Baseline LFM2-VL-450M | Distilled student | Δ |
|---|---:|---:|---:|
| POPE (Overall) | 86.2 | **38.5** | **−47.7** |
| RealWorldQA | 42.0 | 36.0 | −6.0 |
| MMBench | 74.0 | 57.0 | −17.0 |

All three regressed; POPE collapsed.

---

## Diagnosis — not a format break, a grounding regression

The distilled student's answers are **well-formed** ("Yes."/"No."/"A") — it did not lose the answer format. But the answers are **wrong**: POPE recall 33%, precision 45% → it outputs "No" too often, **under-detecting objects**.

So the mechanism is **task interference / catastrophic forgetting**: LoRA fine-tuning on caption-only data at lr 2e-4 for 3 epochs over-wrote the student's instruction-tuned grounding/VQA ability, even though it still emits the right format. The model was pulled toward "produce a fluent description" at the cost of "answer this question correctly."

This connects directly to P2-1.1/P2-1.3: the teacher's edge is on **MCQ/reasoning**, *not* captioning (it's not even a CLIP-score leader). We then distilled **captioning** — the teacher's weaker, task-misaligned skill — and it actively hurt the MCQ metrics we care about.

---

## Why the pilot was worth it

This is exactly what the pilot-first approach is for. For ~3 hours of compute and a 5K cache, we learned that the naive "distill teacher captions" objective **backfires** — *before* spending the full 50K-image cache + 3-seed fine-tune chasing it. A negative result this clear is a cheap, high-value outcome.

---

## Strategy B v2 — what to change

The fix is to **align the distillation data with the target skill (MCQ/grounding), and prevent forgetting**:

1. **Task-aligned teacher outputs.** Have the teacher answer **VQA / yes-no / multiple-choice style** prompts (grounded Q&A), not open captions. Distill the skill we actually measure.
2. **Rehearsal against forgetting.** Mix in instruction-following / original-task examples (or much lower LR, fewer epochs, smaller LoRA) so the student augments rather than overwrites its base ability.
3. **Re-baseline after each change** on the same path (baseline 86.2 POPE is the bar to beat, not regress).

The distillation pipeline (`services/distillation_pipeline.py`) is reusable — only the **prompt / target format** changes (caption → grounded Q&A). The fine-tune + same-path eval harness are unchanged.

---

## Reproduce

```bash
# teacher cache (caption — the approach shown here to NOT work)
python services/distillation_pipeline.py --images datasets/coco_train2017 --limit 5000 \
  --out datasets/caption_cache/qwen25_3b_coco5k.jsonl
python runners/finetune_vlm.py --cache datasets/caption_cache/qwen25_3b_coco5k.jsonl \
  --images datasets/coco_train2017 --out artifacts/students/lfm2vl_distill_pilot_s0 \
  --epochs 3 --seed 0 --batch-size 1 --grad-accum 16
python runners/eval_vlmeval.py --models LFM2-VL-450M LFM2-VL-450M-distill \
  --benchmarks POPE RealWorldQA MMBench_DEV_EN --n-samples 100 --output results/eval_p2_pilot_distill
```

Adapter + results are gitignored (local); the distilled student is registered in `runners/eval_vlmeval.py` as `LFM2-VL-450M-distill`.
