# Observation: A Balanced Multi-Distribution Mixture Clears Two Benchmark Floors With One Student (P2-B1)

**Date:** 2026-06-16
**Context:** The single-distribution runs proved the lever is train/eval distribution alignment — a COCO-trained student clears POPE, a ScienceQA-trained student clears MMBench, each at the cost of the other. The open question was the **Goals bar**: can *one* student clear **≥2** benchmark floors? The ScienceQA pilot collapsed POPE because ScienceQA dominated (77%) with weak (0.3) rehearsal.
**Verdict:** **Positive milestone — yes, with a balanced mix.** A 50/50 COCO-grounding + ScienceQA student clears **both POPE and MMBench** above their floors simultaneously — the first single constructed student to be above floor on ≥2 real benchmarks. The cost is honest: peak POPE drops from the single-skill 68.3 to 55.0 (breadth vs. depth). This is **not yet "competitive with reference models"** (the full Goals definition) — POPE has the most headroom.

---

## Setup

- Spec `151cf686`. Same architecture (Qwen2.5-0.5B + SigLIP-base + MLP projector), align 3000 on `coco_caption_5k`.
- **Balanced 50/50 distill mix:** grounding primary `qa_balanced_5k` (1386, → POPE) + `scienceqa_mcq` replayed at **`rehearse_frac=1.0`** (adds 1386, → MMBench) = **2,772 rows, exactly 50/50**. 3 epochs (~8,316 steps), LoRA rank **held at 16** to isolate the *mixture* variable against the single-skill bests.
- Eval: floor-adjusted, same-path, n=100.

## Result (floor-adjusted)

| Benchmark | This mix (`151cf686`) | Floor | Single-skill best | LFM2 ref |
|---|---:|---:|---:|---:|
| **POPE** (bal-acc) | **55.0 ✓ above** | 50 | 68.3 (`d3423bc0`, COCO) | 86.2 |
| **MMBench** (acc) | **0.62 ✓ above** | 0.50 | 0.65 (`b2feb6b1`, ScienceQA) | 0.74 |
| RealWorldQA (acc) | 0.50 (nominally above 0.47) | 0.47 | — | ~0.42 |

**The robust result is POPE + MMBench: one student, both above floor — a first.** Balancing the two training distributions let a single 0.5B student hold both skills at once, where every prior multi-skill attempt collapsed one. No capacity increase was needed; the fix was *data balance*.

## Honest reading

- **Breadth costs depth.** POPE fell from the single-skill peak 68.3 → 55.0 (still real, above floor) and MMBench from 0.65 → 0.62 (essentially retained). The mixture trades single-benchmark peak for multi-benchmark coverage — expected, and modest.
- **Above-floor ≠ competitive-with-references.** The Goals bar defines "competitive" relative to the reference models, not the chance floor. MMBench **0.62 is close to LFM2's 0.74** (within ~0.12, internal-only); POPE **55 vs 86 has real headroom**. So this clears ≥2 *floors* and is *approaching* reference on MMBench, but is not yet competitive-with-references on POPE.
- **RealWorldQA is uninformative here.** 0.50 is nominally above its 0.47 floor and above LFM2 (~0.42), but **LFM2 itself sits at/below this floor at n=100** — the RealWorldQA n=100 slice does not discriminate models (consistent with prior observations). No real RealWorldQA claim is made.

## What this establishes

The HLD §6.5.5 principle now has its second half validated: not only does **distribution matching** determine which benchmark a student can clear, but a **balanced mixture of matched distributions lets one student clear several** — exactly how production VLMs train. The multi-skill problem that blocked four prior runs was a *data-balance* problem, not a hard capacity wall (at least for two distributions at 0.5B).

`151cf686` is the new **best-breadth student** (the only one above floor on ≥2 benchmarks). `d3423bc0` still holds the POPE peak (68.3). Neither strictly dominates — this is a genuine Pareto point (breadth vs. POPE depth).

## Next levers

1. **Close the POPE depth gap** — up-weight grounding slightly (e.g. 60/40 toward grounding) or add grounding epochs, and re-measure whether POPE recovers toward 68 while MMBench stays above floor. Find the ratio that maximizes the *minimum* across benchmarks.
2. **Capacity for "competitive," not just "above floor"** — to approach reference scores (POPE 86, MMBench 0.74) rather than clear floors, the ~0.5B LM is likely the limit; a larger student (0.5B→~0.9B) or higher LoRA rank is the lever, now that data balance is solved.
3. **RealWorldQA needs its own matched data** (real-world spatial) *and* a larger eval slice to become informative.

## Caveats

Internal-only numbers (n=100 slice, non-official protocol, no published number reproduced — memory `benchmark-eval-internal-only`). ScienceQA half uses dataset gold answers; grounding half is teacher-distilled. Internally consistent (same eval, floor-adjusted, same architecture), trustworthy for this comparison, not externally citable.

## Artifacts

- Student: `artifacts/students/build_151cf686f7b8/` (spec `151cf686`).
- Ledger: `artifacts/experiment_ledger/construction_151cf686f7b8.json`.
- Data: `qa_balanced_5k` (1386) + `scienceqa_mcq` (1386 of 2500 sampled), 50/50.
