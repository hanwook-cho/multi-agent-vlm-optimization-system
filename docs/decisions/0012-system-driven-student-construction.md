# ADR-0012: System-Driven Student Construction — Parameterized Builder, Agent-Driven Search

**Date:** 2026-06-15
**Status:** Accepted
**Context:** Phase 2 — the next move is P2-B1 (assemble a right-sized student from the Qwen2.5-VL-3B lineage and distill). The question is *who* builds it: a human, one-off (Tier-2), or the system.

---

## Context

The deliverable of this project is the **multi-agent VLM optimization system**, not any single model. A competitive model is **proof-of-work** (Goals §5). Two Phase-2 results (P2-D1, P2-D2) already demonstrated the system's *reasoning* loop — propose → test → record → re-route, with no human config-picking (P1). What they did **not** exercise is the system's *construction* ability: both were distillation runs *into an existing* model.

P2-B1 (assemble Qwen2.5-0.5B LM + small SigLIP vision + projector, then distill from the 3B) is classified as **Tier-2** (code, human-implemented per HLD §6.3). If a human hand-builds it as a one-off, the result advances the *model* but tests the *human's* engineering — not the system's autonomy. That is the failure mode this project must avoid: doing model work and calling it system work.

**Directive (user, 2026-06-15):** *"System should do. Not human implement."*

---

## Decision

**Convert model construction from a Tier-2 one-off into a parameterized capability the agent drives.** Concretely:

1. **The human writes one generic, declarative builder — once.** A `build_student` runner consumes a **`StudentSpec`** (which LM, which vision encoder, projector shape, init strategy, alignment recipe, distillation recipe, eval suite) and executes the full pipeline: *assemble VLM → align projector → distill from teacher → same-path eval → emit MetricsReport → ledger.* This irreducible scaffolding (assembling a multimodal forward pass is code) is written **generically**, not for one specific student.

2. **Every *instance* of construction is a config the agent proposes.** A `StudentSpec` is to `build_student` what a quantization config was to the Phase-1 runners: a **Tier-1-style search point**. The Search Strategist proposes `StudentSpec`s, the runner executes them, results land in the ledger, and the agent re-routes on real outcomes. The search space (LM size, vision encoder, projector depth, init=scratch/adapt, distill data recipe, LoRA rank, budget) is the agent's to explore.

3. **The human's role collapses to writing the harness, not the model.** After `build_student` exists, P2-B1 is not "a human builds a model" — it is "the agent runs its first construction experiment." Subsequent students cost the agent a new spec, not the human a new implementation.

This honors P1 (no manual config tweaking) at the level that matters for Phase 2: the system *constructs and searches*, the human *enables*.

---

## Design

### StudentSpec (the agent's new search space)

A declarative, schema-validated config, e.g.:

```jsonc
{
  "lm":        "Qwen2.5-0.5B-Instruct",          // same-family as the 3B teacher
  "vision":    "google/siglip-base-patch16-224", // lean vision budget
  "projector": {"type": "mlp", "depth": 2, "hidden": 2048},
  "init":      "scratch",                          // or "adapt:<existing small VLM>"
  "align":     {"data": "coco_caption_5k", "steps": 2000},     // stage-1: connect modalities
  "distill":   {"teacher": "Qwen2.5-VL-3B",
                "data": "qa_balanced_5k",          // task-aligned + HARD NEGATIVES (P2-D2 fix)
                "lora_r": 16, "epochs": 3, "rehearse_frac": 0.2},
  "eval":      {"benchmarks": ["POPE", "RealWorldQA", "MMBench_DEV_EN"], "n": 100}
}
```

### build_student (human, written once)

`runners/build_student.py` — consumes a `StudentSpec`, runs assemble → align → distill → eval, emits a MetricsReport keyed by spec hash into the experiment ledger. Reuses what already exists: `services/distillation_pipeline.py` (teacher cache), `runners/finetune_vlm.py` (LoRA training), `runners/eval_vlmeval.py` (same-path eval + registry).

### Agent integration

The Search Strategist gains a tool/hypothesis class that emits `StudentSpec`s (not just config tweaks), reads back ledger results, and re-routes — closing the construction loop the way it already closes the distillation loop.

### Carry the P2-D2 lesson into the data layer

The distillation data recipe must support **balanced hard negatives** (yes/no questions about *absent* objects, ~50/50), addressing the presence-bias collapse that broke P2-D2. This is a `distill.data` option the agent can select, not a hardcoded fix.

---

## Consequences

- The one-time human cost is the **generic builder + schema + agent wiring**, not a bespoke model. From then on, model construction is agent-driven search.
- Proof-of-work is still produced — but *through* the system (the agent runs the spec that yields a competitive student), which is what Goals §5 actually requires.
- P2-C1 (hard-prune the 3B) and P2-B1 (assemble) become **two `StudentSpec` families** over the same builder, not two separate human builds — the agent can compare them.
- Risk: the builder is real engineering and must run on the 16GB Mac (batch-1, staged, LoRA). Mitigated by an incremental smoke-first sequence (below).

---

## Plan (sequenced, each step independently verifiable)

- **B1.0 ✅ Generic builder skeleton + smoke.** `schemas/students.py` (`StudentSpec`, content-addressable) + `schemas/student_spec.schema.json` + `runners/build_student.py` assembling a `StudentVLM` (vision + MLP projector + LM, LLaVA-style prepend) and running assemble → align → distill → generate. Smoke verified on the 16GB Mac (SigLIP-base + Qwen2.5-0.5B). *(Human scaffold, once.)*
- **B1.1 ✅ Balanced hard-negative QA recipe.** `distillation_pipeline.py` `--mode qa_balanced` emits grounded ~50/50 present("Yes")/absent("No") presence questions + open Q&A. Pilot: 6Y/6N balanced, grounded. *(The P2-D2 fix, parameterized; full cache is compute-gated.)*
- **B1.2 ✅ Agent drives the builder.** `propose_student` tool on the Search Strategist emits a `StudentSpec`; `services/construction_loop.py` consumes it, runs `build_student`, and writes a `student_construction` ledger entry → next `propose_next()` re-routes. Verified live end-to-end (agent-enqueued P2-B1 spec → real assemble → smoke build → ledger). The construction loop is closed.
- **B1.3 — First real construction run.** The agent runs its first P2-B1 spec (generate the balanced cache, build a real student); same-path MCQ eval vs the LFM2-VL-450M benchmark (bar: POPE ≥ ~86, no regression). Record; the agent proposes the next spec on the outcome.

---

## Open issues

- Init strategy (scratch vs. adapt an existing small VLM) materially changes alignment cost on a 16GB Mac; B1.0 should measure it cheaply before committing.
- Projector/architecture choices are part of the agent's search, but the *space* must be bounded in the schema so proposals stay buildable.
- Eval must keep the inference path constant (P2-1.3 methodology) across every constructed student for valid comparison.
