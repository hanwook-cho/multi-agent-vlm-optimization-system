# Observation: Distribution-Matched Training Clears the MMBench Floor — the Lever Is Train/Eval Distribution Alignment (P2-B1)

**Date:** 2026-06-16
**Context:** Four consecutive task-interference negatives (P2-D1 caption, P2-D2 task-aligned-into-LFM2, P2-B1 MCQ-data, P2-B1 rehearsal+full-epoch) left the constructed student with real POPE grounding (bal-acc 68.3) but stuck **at the MMBench floor**. We hypothesized the cause was **train/eval distribution mismatch**: the student trains on COCO (POPE's distribution), but MMBench is mostly science/knowledge/reasoning — off-distribution for COCO.
**Verdict:** **Positive — hypothesis validated, and bidirectionally.** Training on a distribution-matched source (ScienceQA) **cleared the MMBench floor for the first time** (0.65 vs 0.50 floor; only 0.09 below the LFM2 yardstick's 0.74). The cost: POPE collapsed (rehearsal at this ratio didn't protect it). The student succeeds on whichever benchmark matches its training distribution — proven both directions.

---

## The diagnosis (verified before building)

Two independent checks established the cause was **distribution, not MCQ format or model capacity**:

1. **MMBench is ~70% non-COCO content.** MMBench_DEV_EN (4,329 q) is dominated by `physical_property_reasoning`, `function_reasoning`, `identity_reasoning`, `future_prediction`, `ocr`, `celebrity_recognition`, `social/nature_relation`, `image_style/emotion`, `structuralized_imagetext_understanding`. Sample items are ScienceQA-style (*"which was the independent variable?"*, *"describe the Great Victoria Desert ecosystem"*). The COCO-able perception subset (object/attribute/spatial/scene) is only ~30% → a COCO-trained student cannot clear a 50% floor through COCO alone.
2. **The student already does MCQ format.** The rehearsal student (`55dcb001`) scored **59/100 on in-distribution COCO MCQ** (chance 25%) with clean single-letter outputs, yet floored MMBench. So format works; the gap is distribution.

## Setup

- Spec `b2feb6b1`. Same architecture (Qwen2.5-0.5B + SigLIP-base + MLP projector), align 3000 on `coco_caption_5k` (converged, 1.79).
- **Distill on `scienceqa_mcq`** (2,500 ScienceQA gold MCQ — the closest public match to MMBench's distribution, natively multiple-choice; `runners/build_scienceqa_cache.py`) as **primary**, with **`qa_balanced_5k` grounding replayed at `rehearse_frac=0.3`** to test POPE coexistence. 2 epochs (~6,500 steps, distill loss 0.31).
- Eval: floor-adjusted, same-path, n=100.

## Result (floor-adjusted)

| Training data | POPE (bal-acc) | MMBench (acc) | RealWorldQA |
|---|---:|---:|---:|
| COCO (`d3423bc0`) | **68.3 ✓** | 0.44 (floor) | ~0.41 (floor) |
| **ScienceQA (`b2feb6b1`)** | 34.4 (collapsed) | **0.65 ✓ above floor** | 0.42 (floor) |
| *LFM2 yardstick (ref)* | *86.2* | *0.74* | *~0.42* |

- **MMBench 0.65, above_floor=True** — the **first constructed student to clear a real-benchmark MCQ floor**, and only **0.09 below LFM2** (internal-only). Distribution fix works.
- **POPE collapsed** to bal-acc 34.4 (raw 5.7 → near-constant answer): the 0.3 grounding rehearsal was too weak against 77% ScienceQA. RealWorldQA unchanged at floor (its xAI real-world distribution matches neither COCO nor ScienceQA).

## The lesson — train/eval distribution alignment is the lever

Across all P2-B1 runs, the student scores **above floor exactly on the benchmark whose distribution matches its training data, and at floor otherwise**:

- COCO training → **POPE** (POPE is built on COCO) clears; MMBench floors.
- ScienceQA training → **MMBench** (ScienceQA ≈ MMBench distribution) clears; POPE floors.
- Neither → RealWorldQA (xAI real-world) floors in every run.

This is a clean, bidirectional demonstration that **the dominant lever for a small constructed student is matching the training-data distribution to the target benchmark** — the same reason production VLMs (LLaVA et al.) train on broad mixtures spanning their eval suites. The earlier four "task-interference" negatives were really a *distribution* story: we were training on COCO and evaluating off-distribution.

## What remains — the multi-skill problem

What the system has now: it can produce a student that clears **either** POPE **or** MMBench, by choosing the matched training distribution. What it cannot yet do: clear **both at once** — every attempt to hold two skills has shown interference at this 0.5B capacity / data scale (POPE collapsed here; MMBench never came up in the COCO runs). The Goals success bar ("competitive on ≥2 of {POPE, MMBench, RealWorldQA}") therefore now hinges on the **multi-distribution mixture** problem, not on any single benchmark.

Next levers:
1. **Balanced multi-distribution mixture** — co-train on COCO-grounding + ScienceQA (+ a RealWorldQA-like source) with tuned ratios and stronger rehearsal, and measure whether a single student can clear ≥2 floors. This is the direct path to the Goals bar.
2. **Capacity** — if mixing keeps trading one skill for another, the ~0.5B LM may be too small to hold the distributions simultaneously; a slightly larger student (e.g. 0.5B→0.9B) is the lever.
3. **Two best students stand, per axis:** `d3423bc0` (POPE 68.3) and `b2feb6b1` (MMBench 0.65). Neither dominates the other.

## Caveats

Internal-only numbers (n=100 slice, non-official protocol, no published number reproduced — memory `benchmark-eval-internal-only`). ScienceQA targets are dataset **gold** answers (distribution-matched supervised MCQ), not teacher-distilled — a deliberate choice to test the distribution hypothesis cleanly; the grounding rehearsal half is still teacher-distilled COCO. The comparison is internally consistent (same eval, floor-adjusted, same architecture) and trustworthy for this A/B, not externally citable.

## Artifacts

- Student: `artifacts/students/build_b2feb6b16d7f/` (spec `b2feb6b1`).
- Ledger: `artifacts/experiment_ledger/construction_b2feb6b16d7f.json`.
- Data builder: `runners/build_scienceqa_cache.py` (`scienceqa_mcq`, 2,500 rows).
- In-distribution MCQ pre-check: 59/100 on the `55dcb001` student (COCO MCQ, chance 25%).
