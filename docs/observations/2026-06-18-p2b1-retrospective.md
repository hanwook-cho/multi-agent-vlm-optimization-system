# Retrospective: P2-B1 — System-Driven Student Construction (consolidated)

**Date:** 2026-06-18
**Status:** P2-B1 **CONFIRMED and paused.** This consolidates the full arc and records what is established, what is deferred, and why we are stopping here rather than grinding further at 0.5B.

---

## The one-paragraph result

The system-driven construction loop works: the Search Strategist proposes a content-addressed `StudentSpec`, and a deterministic pipeline assembles (LM + vision + projector), aligns, distills from the Qwen2.5-VL-3B teacher, scores on the same path as the benchmark, and records to the ledger — **with no human in the build.** The central empirical finding is that **train/eval distribution matching is the dominant lever** for a small constructed student: it clears exactly the benchmark whose distribution matches its training data, and floors otherwise. We have **per-axis students with real, measured capability**, but **a single student that robustly clears ≥2 benchmark floors is variance-limited at 0.5B** — a capacity problem, deferred to Phase 3.

## The arc (what each step taught)

| Step | Result | Lesson |
|---|---|---|
| P2-D1 caption distillation | Negative (regressed) | Distilling the teacher's *weak* skill (captions) into LFM2 caused forgetting |
| P2-D2 task-aligned into LFM2 | Negative (presence-bias collapse) | Don't distill *into* an already-tuned edge model; need a right-sized student |
| B1.3 first constructed student | "Degenerate" → **decode bug** | `lm.generate(inputs_embeds=…)` ignored image embeds; floor-adjusted eval needed |
| B1.3 scale-up (`d3423bc0`) | **POPE bal-acc 68.3 (real)**, MMBench at floor | Construction produces real grounding; COCO training → POPE |
| MCQ-data mix (`1507d317`) | Negative (interference) | Adding MCQ regressed POPE, didn't lift MMBench |
| Rehearsal + full-epoch (`55dcb001`) | Negative — **falsified under-training** | Not a budget problem; MCQ-mixing itself collapses grounding |
| **Diagnosis** | MMBench is ~70% science/knowledge (off-COCO); student does MCQ fine (59/100 in-dist) | The gap is **distribution**, not format/capacity |
| ScienceQA (`b2feb6b1`) | **MMBench 0.65 — first above floor** | Distribution matching works (bidirectional proof) |
| 50/50 mix (`151cf686`) | Both floors cleared — *once* | A balanced mixture can give multi-skill… |
| **Seeded variance test** | POPE 5–62 across seeds (2/4); MMBench robust | …but it was variance; **no seeding** was a confound. Fixed. |
| 60/40 grounding-heavy (3 seeds) | POPE median 13→48 but still 1/3 above floor | Ratio shifts POPE's *mean*, not its *variance* → capacity wall |

## What is CONFIRMED

1. **The construction capability** (ADR-0012): the agent constructs and evaluates students autonomously; the loop is the deliverable it was meant to be.
2. **Distribution matching is the lever** (HLD §6.5.5): matched training data reliably delivers the targeted benchmark skill; MMVBench-from-ScienceQA reproduces across seeds.
3. **Per-axis students** with genuine capability: `d3423bc0` (POPE grounding) and `b2feb6b1` (MMBench ~0.65, ~0.09 below the LFM2 yardstick — internal-only).

## What is DEFERRED (to capacity / Phase 3)

A single student **robustly** clearing ≥2 floors. At 0.5B, POPE's yes/no calibration is init-unstable (balanced-acc spread ~40–57 pts across seeds), and the data ratio only shifts the mean. This is a capacity / training-stability limit; the levers (a larger LM that breaks the ≤450M edge budget; or projector warm-start / batch>1) are Phase-3 / budget-relaxation decisions, not Phase-2 data tweaks. The Goals "competitive on ≥2 benchmarks with one model" bar is therefore **gated on capacity, not data** — and is not claimed as met.

## Methodology locked in (the durable process wins)

- **Floor-adjusted, same-path eval** — raw scores are meaningless without the chance/majority floor; validated against a known-good reference (LFM2).
- **Multi-seed reporting** — single-run construction results (especially POPE) are not trustworthy; report median/min over ≥3 seeds. (This caught an over-claimed "milestone.")
- **Verify before building** — the distribution diagnosis was confirmed (MMBench category analysis + in-distribution MCQ probe) *before* spending compute on ScienceQA.

## Why stop here

The multi-skill-at-0.5B problem is now well-characterized: data is solved, variance/capacity is the wall. Further mix-tuning has diminishing returns (the 60/40 run already showed the ratio can't fix variance). The honest, well-supported conclusion — *the system reliably acquires a benchmark's skill by matching its training distribution; robust multi-skill needs more capacity* — is itself the result. P2-B1 is marked **CONFIRMED** and paused; reopen with a larger student (Phase 3) or a training-stability fix.

## Artifacts

- Best students: `artifacts/students/build_d3423bc0…/` (POPE), `…/build_b2feb6b16d7f/` (MMBench).
- Observations: the dated P2-B1 series (2026-06-15 → 2026-06-18) under `docs/observations/`.
- Decision record: [`ADR-0012`](../decisions/0012-system-driven-student-construction.md); principle in [HLD §6.5.5](../VLM_Optimization_HLD.md).
