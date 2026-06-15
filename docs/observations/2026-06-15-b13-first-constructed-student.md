# Observation: First System-Constructed Student (B1.3) — Loop Proven, Student Under-Trained

**Date:** 2026-06-15
**Context:** Phase 2 / ADR-0012 B1.3 — the first end-to-end run of the system's model-CONSTRUCTION loop. The Search Strategist proposed a `StudentSpec` (P2-B1); the construction loop assembled a brand-new VLM, distilled it, and scored it same-path — with no human in the build.
**Verdict:** **Milestone met (the construction loop works end-to-end) — the student itself is degenerate/under-trained, exactly as expected for a deliberately capped first run.** A concrete, actionable signal for the agent's next spec.

---

## What ran (agent → spec → build → distill → eval → ledger)

- **Agent proposed** (`propose_student`): `lm=Qwen/Qwen2.5-0.5B-Instruct`, `vision=google/siglip-base-patch16-224`, MLP projector (depth 2, hidden 2048), `distill.data=qa_balanced_5k`. Spec `df64c49b`.
- **Built** by `services/construction_loop.py` → `runners/build_student.py`: assembled `StudentVLM` (SigLIP vision + fresh projector + Qwen LM, LLaVA-style prepend), **align 200 steps** (projector-only) + **distill 1000 steps** (LoRA r=16 + projector) on the **balanced cache** (481 images, 462Y/462N presence + 399 open, B1.1).
- **Scored same-path** (`runners/eval_student.py`, n=100) on POPE / RealWorldQA / MMBench, then **recorded** to `artifacts/experiment_ledger/construction_df64c49bd7ef.json`.

All stages ran on the 16GB Mac, batch-1, no swap-thrash.

---

## Result (honest)

| Signal | Value |
|---|---|
| align final loss | **2.378** (≈ unchanged from start) |
| distill final loss | 0.765 (fell — but on un-aligned vision) |
| POPE | **Overall null** (predictions unparseable / degenerate) |
| RealWorldQA | **0.0** (Δ −0.42 vs LFM2 0.42) |
| MMBench | **0.0** (Δ −0.74 vs LFM2 0.74) |
| greedy generations | gibberish (e.g. repeated tokens, wrong-language fragments) |

The student is **non-functional** — it scores ~0 everywhere and emits gibberish.

---

## Diagnosis — alignment never converged

The tell is **align loss stayed at ~2.38** over 200 steps. A *freshly-initialised* projector has to learn the entire SigLIP→Qwen embedding mapping from scratch; 200 steps on ~480 images is nowhere near enough. So the projected "image tokens" are effectively **noise** prepended to the prompt, and the LM — never having seen useful vision tokens — produces garbage. The distill loss falling to 0.77 is misleading: the LoRA learned token-level patterns of the short answers without any grounding, because the vision pathway carried no signal.

This is **not** a flaw in the loop or the data (the balanced cache is correct, B1.1). It is a **budget/curriculum** problem: stage-1 alignment must actually converge before stage-2 distillation can teach grounding.

---

## What the system does next (re-route)

P2-B1 stays **OPEN** with this result in its `result_summary`, so the Search Strategist proposes a **refined spec**, not a different hypothesis. Concrete levers the first run points to:

1. **Far more alignment** — projector convergence is the gate. 200 → several thousand steps (and watch align loss actually drop), on a much larger caption set (align uses `coco_caption_5k`, which has 5000 — the cap to 481 starved it).
2. **Larger / uncapped data** — 481 images is tiny; scale the balanced cache toward the planned 5k.
3. **Consider `init=adapt:<small VLM>`** — warm-starting from an existing small VLM's projector/alignment sidesteps the cold-start, instead of `scratch`.

These are spec edits the agent can make — which is the point of ADR-0012: the first run produced a real, recorded, diagnostic result, and the loop refines from here.

---

## Why this is the milestone, not the disappointment

For the first time the **system built a model end-to-end from a proposal** — assemble, align, distill, score same-path, record to the ledger — with no human implementing the build. The student is bad, but it is a *real, measured, reproducible* student that the agent can now improve by proposing a better spec. That closed construction loop is the Phase-2 deliverable (the competitive model is downstream proof-of-work).

---

## Artifacts

- Ledger: `artifacts/experiment_ledger/construction_df64c49bd7ef.json`
- Eval: `artifacts/students/build_df64c49bd7ef/eval/student_{POPE,RealWorldQA,MMBench_DEV_EN}.json`
- Spec: P2-B1 `df64c49b` (agent-proposed)
- Balanced cache (B1.1): `datasets/caption_cache/qwen25_3b_qa_balanced5k.jsonl` (481 imgs, 462Y/462N + 399 open)
