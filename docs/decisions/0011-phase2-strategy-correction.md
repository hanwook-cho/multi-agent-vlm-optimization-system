# ADR-0011: Phase 2 Strategy Correction — LFM2 is the Benchmark, Not a Student

**Date:** 2026-06-13
**Status:** Accepted
**Context:** Phase 2 — produce an edge model from Qwen2.5-VL-3B competitive with the small-edge VLMs.

---

## Context

Phase 2's central claim (Goals §5, criterion S3): *"the starting point of the optimization was **not** an already-edge-optimized model. The Phase 2 starting point is Qwen2.5-VL-3B."* The deliverable is a ~450M-class edge model, **derived from the open general-purpose Qwen2.5-VL-3B**, that is competitive with the LFM2-VL-450M / SmolVLM-500M / MiniCPM-V / FastVLM **benchmarks**.

The `DetailedPlan_Phase2.md` I wrote (and the first distillation pilot) instead used **LFM2-VL-450M as the distillation student** — fine-tuning teacher outputs *into* an already-edge-optimized model. This was wrong.

---

## What went wrong, with evidence

1. **It violates the Phase 2 premise (S3).** LFM2-VL-450M is the product of a year+ of expert optimization. Using it as the student means the optimization "started from" an already-optimized model — the exact thing Phase 2 forbids. Any quality of the result couldn't be attributed to the system vs. Liquid AI's prior work (provenance failure).

2. **It conflates Phase 2 with Phase 3.** Goals §4/§5 reserve "squeeze more from the already-optimized LFM2-VL-450M" for **Phase 3** (the complementary "useful on top of expert optimization" claim). Distilling into LFM2 is a Phase 3 question.

3. **It failed empirically.** The pilot (caption-only distillation into LFM2-VL-450M) **regressed on every MCQ benchmark**, same-path: POPE 86.2→38.5, RealWorldQA 42→36, MMBench 74→57 (observation `2026-06-13-distill-pilot-caption-only-regresses.md`). You can't meaningfully improve an already-optimized model this way.

4. **The architecture budget makes "compress the 3B" non-trivial too.** Measured param breakdown of Qwen2.5-VL-3B: vision encoder **668.7M** + embeddings/lm_head **311.2M** = **980M alone — already > 2× the 450M target**, before any LM layers. Pruning LM layers cannot reach the budget; the vision encoder must be replaced and the 152K-vocab embedding shrunk.

---

## Decision

1. **LFM2-VL-450M is the BENCHMARK / yardstick** for Phase 2 (size + performance bar: same-path POPE 86.2, RWQA 42, MMBench 74). It is **never a student to train.**
2. **The Phase 2 edge model's lineage must be Qwen2.5-VL-3B** — either by compressing the 3B itself (prune + distill-recover, P2-C1) or assembling a right-sized open student and distilling from the 3B (P2-B1). Given the architecture budget, P2-C1 collapses toward P2-B1.
3. **The distillation objective must be task-aligned** to the measured skill (grounded VQA/MCQ), not captioning (P2-D2), with rehearsal against forgetting.
4. **The system chooses the approach, not a human.** The candidate approaches (P2-D1 tried/regressed, P2-D2, P2-C1, P2-B1) live in the Search Strategist's hypothesis table; the agent proposes/sequences them (it proposed P2-D2 after reading the P2-D1 regression). This preserves criterion P1 (no manual config tweaking). The current LFM2-based runs are **method validation** (fast loop to test task-aligned distillation), explicitly not the final deliverable.

---

## Consequences

- `DetailedPlan_Phase2.md` Strategy B (distill into LFM2/SmolVLM) is **demoted/removed** from Phase 2; the LFM2 squeeze moves to Phase 3.
- Primary Phase 2 approach becomes **derive a ~450M student from Qwen2.5-VL-3B** (P2-B1 assemble-small-student is the most tractable; P2-C1 hard-prune as comparison).
- The distillation pipeline + fine-tune + same-path eval built during the pilot are **reusable** for the corrected approach — only the student definition and the distillation target (caption → grounded Q&A) change.
- All cross-model quality comparisons hold the inference path constant (ADR-pending P2-1.3 methodology).

## Open issues

- The LFM2-based P2-D2 run validates the *method* (task-aligned distillation + rehearsal); a positive result there does not satisfy Phase 2 — the student must still be re-based on the 3B (P2-B1).
- Assembling the P2-B1 student (Qwen2.5-0.5B LM + small SigLIP vision) is a Tier-2 build the agent has flagged; sequencing it is the next system decision.
