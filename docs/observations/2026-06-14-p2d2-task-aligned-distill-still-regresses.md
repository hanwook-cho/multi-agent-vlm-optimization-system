# Observation: Task-Aligned Distillation + Rehearsal Still Regresses the Student (P2-D2)

**Date:** 2026-06-14
**Context:** Phase 2 P2-D2 — the Search Strategist's proposed fix for the P2-D1 caption-only regression. Distill the *measured* skill (grounded Q&A) instead of captions, and mix caption rehearsal to prevent forgetting.
**Verdict:** **Negative — task-aligned Q&A distillation + rehearsal still regressed the student on every MCQ benchmark.** The POPE failure mode *flipped sign* (now an always-"Yes" collapse), exposing a data-balance defect in naive teacher-generated Q&A. Confirms the D-series (distill INTO LFM2) is exhausted; pivot to P2-B1.

---

## Setup

- **Teacher:** Qwen2.5-VL-3B (fp16) generated **11,221 grounded Q&A pairs** over 5,000 COCO train2017 images (`--mode qa`: 3 diverse Q&A per image, including ≥1 yes/no "is object X present" question).
- **Student:** LFM2-VL-450M, LoRA (r=16, α=32, q/v/o_proj, 933,888 params = 0.21%), 3 epochs, lr 2e-4, batch-1, grad-accum 16, seed 0.
- **Rehearsal:** 20% caption mix-in from the P2-D1 5K caption cache → 13,465 total training records. Final `train_loss` 4.543 (descended 13.4 → ~4.2, plateaued early). Runtime 6.7 h.
- **Eval:** POPE / RealWorldQA / MMBench, 100-sample slices, **same fp16 transformers path** for baseline and student (P2-1.3 methodology), baseline re-run in the same session.

---

## Result

| Benchmark | Baseline LFM2-VL-450M | P2-D2 distilled | Δ | P2-D1 caption Δ (for ref) |
|---|---:|---:|---:|---:|
| POPE (Overall) | 87.7 | **66.7** | **−21.0** | −47.7 |
| RealWorldQA | 42.0 | **37.0** | −5.0 | −6.0 |
| MMBench | 74.0 | **51.0** | **−23.0** | −17.0 |

Less catastrophic than caption-only on POPE, but **worse on MMBench**. Net: still regressed on all three. **Did not fix it.**

---

## Diagnosis — the failure mode flipped: presence-bias collapse

POPE detail for the distilled student: **acc 50.0, precision 50.0, recall 100.0.** On POPE's balanced yes/no set, answering "Yes" to everything yields exactly this signature. The student collapsed to an **always-"Yes" / always-present** prior.

This is the **opposite** of P2-D1, which over-answered "No" (recall 33%, under-detection). Same regression, mirror-image bias.

**Root cause is the teacher Q&A data, not generic forgetting.** The QA-generation prompt asked for "a yes/no question about whether a specific object **is present**." The teacher naturally asked about objects it actually saw in the image — so the overwhelming majority of yes/no targets had answer **"Yes."** The student learned a presence-bias prior and now confirms every object. Naive teacher-generated Q&A is **data-imbalanced**: it lacks **hard negatives** (questions about absent objects), which is exactly what POPE tests.

---

## The deeper lesson — the D-series is exhausted

Both P2-D1 and P2-D2 distilled *into* LFM2-VL-450M, and both regressed. That is consistent with the ADR-0011 correction: **LFM2 is already edge-optimized** (it is the *benchmark*). Any LoRA perturbation — caption or Q&A, with or without rehearsal — moves it **off its tuned optimum**. Distillation can only *add capability to a model that lacks it*; it cannot improve a model already at its task optimum without a teacher that is itself better on that task (and on MCQ the 3B teacher's margin is modest).

So no amount of D-series tuning (distill INTO LFM2) will *beat* the benchmark. The method is sound; the **base is wrong**.

---

## What the system does next

Feeding this back to the Search Strategist (hypothesis P2-D2 → **REGRESSED**). Two consequences encoded:

1. **Pivot to P2-B1** — assemble a right-sized student (Qwen2.5-0.5B LM + small SigLIP vision) and distill from the 3B. There, distillation *adds* capability to an under-trained small model instead of overwriting a tuned one. This is the corrected primary path (ADR-0011) and the eventual deliverable must derive from the 3B, not LFM2.
2. **If any future distillation reuses teacher Q&A**, the data pipeline must **balance hard negatives** (generate absent-object yes/no questions, ~50/50) before the targets can teach grounding rather than a presence prior.

---

## Why these two pilots were worth it

For ~10 h of local compute we established, with same-path rigor, that **distilling into the benchmark is a dead end in two independent ways** (caption-misalignment *and* base-already-optimal), and we discovered a concrete data defect (Q&A imbalance) that would have silently poisoned any larger run. The agent proposed P2-D2 from the P2-D1 result; the negative result now routes it to P2-B1. That loop — propose, test, record, re-route — is the multi-agent optimization system working as intended.

---

## Artifacts

- Adapter: `artifacts/students/lfm2vl_qa_distill_s0/adapter/`
- QA cache: `datasets/caption_cache/qwen25_3b_qa5k.jsonl` (11,221 pairs)
- Eval JSONs: `results/phase2_p2d2_eval/` (baseline + distill, POPE/RWQA/MMBench)
- Hypothesis: `agents/search_strategist.py` → P2-D2 (REGRESSED)
- Supersedes the method question raised in [2026-06-13-distill-pilot-caption-only-regresses.md](2026-06-13-distill-pilot-caption-only-regresses.md)
