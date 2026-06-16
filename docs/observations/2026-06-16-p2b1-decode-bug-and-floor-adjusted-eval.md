# Observation: The "Degenerate" Students Were a Decode Bug — and Floor-Adjusting the Eval (P2-B1)

**Date:** 2026-06-16
**Context:** Phase 2 P2-B1. The B1.3 and scale-up constructed students both evaluated as **degenerate (all-zero)**. This is the investigation into *why*, and the correction of two measurement bugs plus an eval-methodology gap.
**Verdict:** **The constructed students were never degenerate — they were mis-measured.** A broken decode path made every score 0; fixing it (and floor-adjusting the metrics) shows the scale-up student has **real visual-grounding signal on POPE (balanced-accuracy 68.3 vs 50 chance) and no demonstrated multiple-choice ability.** "Competitive" was an over-claim; this is the honest, floor-validated result.

---

## Bug 1 — the decoder ignored the image (every score was 0)

`StudentVLM.generate` used `self.lm.generate(inputs_embeds=...)`. For this prepend-an-image-embed setup, HF `generate` does **not** condition on the passed `inputs_embeds` — it emitted a **constant token (`athlon`) regardless of image or prompt**. So B1.3 and the scale-up scored 0 not because the models were bad but because the decoder was broken.

Proof (scale-up student, teacher-forced vs free generation):

| Prompt | Target | teacher-forced argmax | `lm.generate` | forward-based greedy (fix) |
|---|---|---|---|---|
| "Is there a tennis racket?" | Yes | ` Yes` | `athlon` | `Yes` |
| "Is there a traffic light?" | No | ` No` | `athlon` | `No` |

The forward pass was correct all along (training loss converged for a reason). **Fix:** replace `lm.generate(inputs_embeds=...)` with a forward-based KV-cached greedy loop (`StudentVLM.generate`).

## Bug 2 — a redundant POPE suffix collapsed recall

`infer` appended the harness suffix to *every* prompt. But the POPE dataset question **already ends with** "Please answer yes or no." — so the harness added a **second** instruction ("Please answer Yes or No only."). The raw-trained student is fragile to that out-of-distribution double-instruction and shifts toward "No" (recall 40→13). MCQ questions, by contrast, carry **no** answer instruction, so their suffix is needed (without it the student emits an empty string). **Fix:** add the suffix only for MCQ.

## Methodology gap — raw scores are meaningless without the chance floor

Even after the decode fix, the raw numbers misled me. Running trivial constant baselines through the **same scorer** reveals the floors:

| Benchmark | trivial floor | student | LFM2 (known-good) | test discriminates? |
|---|---|---|---|---|
| **POPE** | balanced-acc 50 | **68.3** | 86.2 | **yes** (LFM2 ≫ floor) |
| MMBench | always-B = 0.50 | 0.44 | 0.74 | yes (LFM2 ≫ floor) |
| RealWorldQA | always-B = 0.47 | 0.41 | **0.42** | **no** — even LFM2 is at the floor |

Two things fall out:
1. **The eval *code* is sound.** A known-good model (LFM2 86.2 POPE, 0.74 MMBench) scores far above the chance floor — only possible if the scorer, image↔ground-truth alignment, and slicing are all correct. Validated by a reference model, not asserted.
2. **RealWorldQA at n=100 is uninformative** — even LFM2 sits at the majority floor (0.42 vs 0.47), so *no* model's RWQA number this slice can be trusted. Not a code bug; a statistical-power/slice issue.

Also: for POPE the harness "Overall" is **F1**, which is deceptive on this imbalanced 70/30 slice (always-Yes scores F1 66.7 > the student's 55.8). The trustworthy metric is **balanced accuracy** (student 68.3 vs 50 floor).

## The honest, floor-adjusted result for the scale-up student

- **POPE: real signal** — balanced-accuracy **68.3** vs chance 50 (precision 92, recall 40). Genuine visual grounding, well below LFM2 (86.2) but clearly above floor.
- **MMBench: no signal** — 0.44 is *at* the 0.50 majority floor.
- **RealWorldQA: inconclusive** — the test is uninformative at n=100.

So the constructed student shows **real ability on exactly one axis (POPE grounding)** and **no demonstrated MCQ ability** — consistent with its training data (yes/no presence + open Q&A; no multiple-choice). This is the first trustworthy evidence the construction loop produces a model with *some* real capability — not "competitive," but real.

## What changed in the code

- `runners/build_student.py`: `StudentVLM.generate` → forward-based KV-cached greedy decode; `infer` → raw question + MCQ-only suffix.
- `runners/eval_student.py`: report the **chance floor** (trivial constant baselines via the same scorer) and the **trustworthy metric** per benchmark (balanced accuracy for POPE, accuracy for MCQ), with an `above_floor` flag. Raw scores are never reported without their floor again.

## Process lessons (so this doesn't recur)

1. **Validate the decoder, not just the loss.** Converged training loss + garbage generation = a decode/inference bug, not a training failure. Probe teacher-forced vs free generation.
2. **Always floor-adjust.** A model at chance looks "competitive" until you print the majority/random floor next to it. Bake the floor into the eval.
3. **Validate the test with a known-good model.** If a strong reference doesn't beat the floor, the test (or slice) is uninformative — discard it.
4. **Use the right metric.** F1 "Overall" is misleading on imbalanced slices; use balanced accuracy.

## Supersedes

- [2026-06-15-b13-first-constructed-student.md](2026-06-15-b13-first-constructed-student.md) — its "degenerate / alignment never converged" conclusion was partly the decode bug; B1.3's eval was also mis-measured.
- Any "scale-up is competitive on RealWorldQA / MMBench" framing from this session — retracted; those benchmarks are at-floor / uninformative.

## Artifacts

- Student: `artifacts/students/build_d3423bc0155b/` (Qwen2.5-0.5B + SigLIP, align 3000 + distill 1500).
- Floor-adjusted eval: `results/p2b1_floored/student_*.json` (with `chance_floor`, `trustworthy_metric`, `above_floor`).
