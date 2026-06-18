# Phase 2 Writeup — A Multi-Agent System That Constructs Edge VLMs

*Consolidated narrative of the Phase-2 arc (2026-06). Draws together the dated
observations, ADRs, and STATUS log into one account. Intended as the basis for an
external writeup (blog / preprint).*

> **Thesis.** The deliverable is the **system** — an autonomous multi-agent loop that
> optimizes vision-language models for edge devices. A competitive edge model is the
> *proof-of-work*, not the product. Phase 2 asked: can the system, with a human
> gating only consequential decisions, take a general 3B VLM down to a deployable
> edge student — and what does it learn doing so?

---

## TL;DR

- The system **constructs** a right-sized student autonomously (agent proposes a
  content-addressed `StudentSpec` → assemble SigLIP + projector + Qwen2.5-0.5B →
  align → distill from a 3B teacher → floor-adjusted same-path eval → ledger →
  re-route). No human in the build.
- The **central finding** is a clean, bidirectional result: **train/eval
  distribution matching is the dominant lever** for a small constructed student. It
  clears exactly the benchmark whose distribution matches its training data
  (COCO→POPE, ScienceQA→MMBench) and floors otherwise.
- The student is **edge-viable**: ported to MLX, verified to faithfully reproduce
  the evaluated model, and runs at **~80 tok/s in a ~1.6 GB footprint** on Apple
  Silicon — well inside the iPhone 16 Pro budget.
- **Honest boundaries:** quality is **below the reference models** and **per-axis /
  variance-limited** at 0.5B — not a single competitive multi-skill model. All
  benchmark numbers are **internal-only** (100-sample slices, non-official protocol).

The value is the **method and the system**, plus a genuinely useful negative-result
trail — not a state-of-the-art model.

## The approach

Phase 1 produced reference baselines and a quantization/runtime methodology. Phase 2
pivoted (ADR-0011) away from distilling *into* an already-optimized edge model
(which regressed) toward **constructing** a right-sized student from the
Qwen2.5-VL-3B lineage, and then (ADR-0012) made that construction a **system
capability**: the Search Strategist proposes a declarative spec; a deterministic
loop builds, trains, evaluates, and records it; the result re-routes the next
proposal. The human writes the generic builder once; the agent drives every
instance.

## The central finding — distribution matching is the lever

Four consecutive attempts to give the student a second skill failed as
"task interference":

| Attempt | Result |
|---|---|
| Caption distillation (P2-D1) | regressed grounding |
| Task-aligned-into-LFM2 (P2-D2) | presence-bias collapse |
| MCQ-data mixed in | POPE regressed, MMBench stayed at floor |
| Rehearsal + full-epoch | falsified "under-training" — same collapse |

Diagnosis (verified *before* spending compute): the student floored **MMBench**
because MMBench is ~70% science/knowledge/reasoning — **off-distribution for COCO**
(POPE's distribution), which is all the student trained on. A probe confirmed the
student *could* do multiple-choice (59/100 on in-distribution COCO MCQ) yet floored
MMBench — so the gap was **data distribution, not format or capacity.**

Building **ScienceQA** training data (MMBench's distribution, natively MCQ) then
cleared the MMBench floor for the first time (0.65). Combined with the COCO→POPE
result, the lever is proven **bidirectionally**: a small student acquires exactly
the skill whose distribution it trains on. The four "interference" negatives were,
underneath, a distribution-mismatch story.

## The multi-skill boundary (and a methodology catch)

Can one student hold both? A balanced 50/50 mix cleared **both** POPE and MMBench
above floor — once. But a **seeded variance test** (after we discovered construction
runs had **no RNG seeding** — a real confound) showed the truth: **MMBench is robust
across seeds (0.59–0.70), but POPE is a coin-flip (bal-acc 5–62, ~half above
floor).** So "one student, two skills" was a favorable draw, not a solved property.
A grounding-heavier mix shifted POPE's *mean* up but not its *variance*. Conclusion:
the multi-skill blocker at 0.5B is **capacity / training-stability, not data** —
deferred to a larger student (Phase 3).

This catch is itself a contribution: **single-run construction results are not
trustworthy; claims must be multi-seed.**

## Edge-viability (ADR-0014)

The constructed student had only ever been scored on the Mac. To validate the
*edge* half, we ported it to MLX (the deployment runtime) in Python and **verified
parity before any device work**: the MLX SigLIP matches transformers to
`max|Δ|=2e-04`, and the assembled MLX student produces identical greedy output to
the PyTorch student. At that point Apple-Silicon perf (M4): **TTFT 118 ms, ~80 tok/s,
~1.6 GB peak** — fitting the iPhone budget that made the 3B teacher (6.5 GB)
non-viable. The MLX student *is* the evaluated student, and it is edge-deployable.

## What this is — and is not

**Is:** a working autonomous construction system; a clean, transferable principle
(match training distribution to the eval distribution — what production VLMs do); a
disciplined methodology; an edge-viable, faithfully-reproduced student; a documented
trail of honest negatives that drove every pivot.

**Is not:** a state-of-the-art model. The student is **below the reference models on
quality**, is **per-axis** (different students clear POPE vs MMBench), and POPE is
**variance-limited** at 0.5B. A single competitive multi-skill model is gated on
capacity (Phase 3), not on the method.

## Methodology (the discipline that made the results trustworthy)

- **Floor-adjusted, same-path eval** — raw scores are meaningless without the
  chance/majority floor; validated against a known-good reference.
- **Multi-seed reporting** — single runs (especially POPE) are not trustworthy.
- **Verify before building** — the distribution diagnosis was confirmed before
  spending compute on ScienceQA; the MLX parity gate was checked before any Swift.
- **Honest negatives** — every regression is recorded with its diagnosis; several
  shiny single-run "milestones" were retracted after scrutiny.

## Limitations & future work

- **Benchmark numbers are internal-only** (100-sample slices, non-official protocol;
  no published number reproduced). They are internally consistent and trustworthy
  for steering, **not externally citable**. *Before any external/quantitative claim,
  a full-set official-protocol validation run is required* (deferred to date).
- **Phase 3 — capacity:** a larger student to address robust multi-skill and close
  the quality gap to references.
- **Mode B — research-analyst:** the system's literature-driven exploration half
  (the designed novelty vs plain AutoML) has not yet been exercised.
- **On-device confirmation:** an actual iPhone number (MLX-Swift) and a
  fair quantized perf comparison vs references remain optional follow-ups.

## Pointers

- Decisions: [ADR-0011](decisions/0011-phase2-strategy-correction.md),
  [ADR-0012](decisions/0012-system-driven-student-construction.md),
  [ADR-0014](decisions/0014-on-device-deployment-via-mlx-swift.md).
- Principle: [HLD §6.5.5](VLM_Optimization_HLD.md).
- Detailed results: the dated `observations/` series (2026-06-13 → 2026-06-18),
  esp. the [P2-B1 retrospective](observations/2026-06-18-p2b1-retrospective.md) and
  the [on-device MLX validation](observations/2026-06-18-on-device-mlx-validation.md).
- Primers: [Model Optimization](guide/) · [KV Cache Optimization](guide-kv-cache/).
