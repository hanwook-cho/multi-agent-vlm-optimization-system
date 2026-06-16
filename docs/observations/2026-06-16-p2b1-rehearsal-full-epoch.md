# Observation: Rehearsal + Full-Epoch Training Did NOT Protect POPE — MCQ-Mixing Itself Is the Cause (P2-B1)

**Date:** 2026-06-16
**Context:** The prior MCQ attempt (`1507d317`) regressed POPE while failing to lift MMBench, and was hypothesized to be **under-trained** (1500 steps ≈ 0.8 epoch on the 1877-row mixed set) with **inert rehearsal** (`rehearse_frac` was never wired into the construction path). This run fixes both: rehearsal is now genuinely wired, grounding is kept the primary skill, and the model trains 3 full epochs.
**Verdict:** **Negative — and it falsifies the under-training hypothesis.** With proper rehearsal and ~3.7× the budget, POPE still collapsed and MMBench still sat at the floor. The cross-run comparison isolates **MCQ-mixing itself** (not budget, not rehearsal) as the cause of the grounding collapse.

---

## Setup

- Spec `55dcb001`. Same architecture as the best student (Qwen2.5-0.5B LM + SigLIP-base + fresh MLP projector), align 3000 on `coco_caption_5k` (converged, align loss 1.94).
- **Distill:** primary `qa_balanced_5k` (1386 grounding rows — the skill that yields POPE bal-acc 68.3) with **`mcq` (491) replayed in at `rehearse_frac=0.5`** → reproduces the same validated 1386:491 composition as the prior run, **but trained 3 full epochs (~5631 steps)** instead of ~0.8 epoch. Distill final loss 0.10.
- Rehearsal is now actually applied (it was a no-op in the construction path before this experiment — see the wiring commit). Deterministic mix, batch-1.
- Eval: floor-adjusted, same-path, n=100.

## Result (floor-adjusted)

| Run | Distill data | Steps | POPE (bal-acc) | MMBench | RealWorldQA |
|---|---|---:|---:|---:|---:|
| Best no-MCQ (`d3423bc0`) | grounding only | 1500 (~1 ep) | **68.3 ✓** | 0.44 (floor) | ~0.41 |
| MCQ-data (`1507d317`) | grounding+MCQ (combined) | 1500 (~0.8 ep) | 48.3 | 0.50 (floor) | 0.43 |
| **This (`55dcb001`)** | grounding + MCQ rehearsal | **5631 (3 ep)** | **47.5** | 0.47 (floor) | 0.44 |

(POPE raw Overall was 63.6 but **balanced-accuracy 47.5 < 50 floor** — the yes/no predictions are biased/uncorrelated with truth, i.e. no demonstrated grounding. RealWorldQA 0.44 < 0.47 floor; MMBench 0.47 < 0.50 floor.)

## Why this is decisive

Both MCQ-containing runs land at **~48 POPE regardless of training budget** (0.8 epoch → 48.3; 3 epochs → 47.5), while the no-MCQ run holds **68.3**. The only thing the collapsed runs share is the **presence of MCQ-format data** in the distill mix. Therefore:

- **The interference is not a budget problem.** 3.7× the steps did not rescue POPE — if anything marginally worse.
- **Rehearsal weighting did not protect the prior skill** at this capacity/data scale, even with grounding as the dominant (74%) primary and replayed across full epochs.
- **MMBench never cleared its floor** in any MCQ run — the 491 COCO-object MCQs don't teach MMBench's reasoning/knowledge distribution, with or without more passes.

The under-training hypothesis (the stated reason to run this) is **falsified**.

## What the system does next

P2-B1 stays **IN_PROGRESS**, but the lever space has narrowed sharply. Ruled out as-is: caption distillation (P2-D1), task-aligned-into-LFM2 (P2-D2), MCQ-as-mixed-distill at this budget (`1507d317`), and now **rehearsal/more-epochs to protect grounding while adding MCQ** (`55dcb001`). Remaining levers:

1. **Distribution-matched MCQ** — generate MMBench-like reasoning/knowledge MCQs (not COCO-object presence MCQs) and test whether *in-distribution* MCQ both transfers to MMBench and coexists with grounding. This is the last untested form of the "add MCQ" idea.
2. **Accept single-skill** — conclude that a ~0.5B-class student at this data scale cannot hold POPE grounding *and* learn MCQ from this teacher/data, ship the grounding student (`d3423bc0`), and choose a success target the training distribution actually supports (object-presence grounding), treating MMBench/MCQ as out of reach for this configuration.

**Best student remains the no-MCQ scale-up `d3423bc0`** (POPE bal-acc 68.3 real; MMBench at floor). This run did **not** produce a better student.

## Caveats

Internal-only numbers (n=100 slice, non-official protocol, no published number reproduced — see memory `benchmark-eval-internal-only`). The comparison here is internally consistent (same eval, floor-adjusted, same architecture) and trustworthy for this A/B, not externally citable.

## Artifacts

- Student: `artifacts/students/build_55dcb001a24d/` (spec `55dcb001`).
- Ledger: `artifacts/experiment_ledger/construction_55dcb001a24d.json`.
- Rehearsal wiring + tests: commit landing `runners/build_student.py` `_load_rows` rehearsal + `tests/test_rehearsal_mix.py`.
