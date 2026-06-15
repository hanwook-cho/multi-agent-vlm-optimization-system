# Multi-Agent VLM Optimization System — High-Level Design

*Zero-based derivation of the system architecture from the ultimate goal. Companion to "Multi-Agent VLM Optimization System — Goals." The Goals document defines what we're trying to achieve; this document derives the system that achieves it. A subsequent Detailed Plan document will cover phase-by-phase execution.*

**Audience.** Project owner, future collaborators, external reviewers.
**Status.** Draft v1 + Amendment A (§6.5).
**Last updated.** June 15, 2026 (Amendment A — system-driven construction, per ADR-0012).

---

## 1. How to read this document

This HLD is written **zero-based**. That means I do not start from the five-agent topology in earlier drafts of the build plan; instead, I derive what components the system needs from the goals themselves and arrive at whatever topology that derivation produces.

The document is structured to make the derivation visible:

1. First, the **intrinsic structure of the problem** — what VLM optimization for on-device deployment actually requires, expressed without reference to any solution shape.
2. Then, **what needs autonomous reasoning vs. what's deterministic** — the central cut that separates "agents" from "services."
3. Then, **the two operating modes** — the known-techniques loop (the system in its straightforward mode) and the research-driven escalation (the system when known techniques are exhausted).
4. Then, **where humans gate** — derived from a single principle, applied to the operating modes.
5. Then, the **proposed component topology** — what falls out of the above.
6. Then, **coordination, failure modes, and the human implementation environment** — the parts that distinguish this from generic ML-ops.
7. Finally, **implications for the Goals document** — places where writing the HLD surfaced issues with the goals themselves.

Some sections will reach conclusions that differ from earlier drafts. I'll flag those explicitly when they happen.

---

## 2. What VLM optimization for on-device deployment actually requires

Stripped of any solution shape, the problem is:

> Given a vision-language task and a target edge device, find a deployment configuration — model architecture, weights, compression scheme, runtime backend, inference parameters — that lies on the Pareto frontier of (task quality, latency, memory, on-disk size) when measured on the actual target device.

This problem has intrinsic structural properties that any solution must respect:

**P1. The search space is combinatorially large.** Architecture choices, compression schemes, runtimes, resolutions, vision-token budgets, and decode strategies multiply out to tens of thousands of configurations. Exhaustive search is impossible. Intelligent sampling is required.

**P2. The search space is also continuously expanding.** New techniques are published monthly: new quantization schemes, new vision encoders, new attention patterns. A static search space goes stale within months. The system must be able to incorporate new techniques without re-architecting.

**P3. Real-device measurement is the only ground truth for the deployment-side metrics.** Latency, memory, and energy depend on hardware-specific factors (ANE op coverage, MLX kernel performance, cache behavior, thermal throttling) that no synthetic proxy captures reliably.

**P4. Quality measurement requires careful evaluation infrastructure.** VLM evaluation is notoriously noisy: caption metrics correlate poorly with human judgment, VQA benchmarks have prompt-sensitivity, retrieval quality depends on dataset composition. A naive "did the model get the right answer" check is not enough.

**P5. Most candidates are bad.** The vast majority of configurations sampled from the search space will be strictly dominated by existing points on the Pareto frontier. Spending real-device measurement budget on obviously-bad candidates is wasteful. Cheap filters must precede expensive measurement.

**P6. Training is expensive and asymmetric.** Fine-tuning a 450M VLM takes hours on the Compute Mac (M4 16 GB initially; M5 Pro 32 GB when available). Training-required improvements ("try fine-tuning with this new technique") cost much more than training-free improvements ("try this quantization scheme on existing weights"). The search must distinguish them.

**P7. The optimization is iterative and stateful.** Each round of experiments produces results that should inform the next round. A system that runs independent batches and forgets is dramatically less efficient than one that maintains and updates beliefs about what works.

**P8. Some actions are irreversible.** Deploying a model to a real device, spending significant compute, or changing what the evaluation measures all have consequences that can't be cheaply undone. These actions warrant deliberation before execution.

**P9. The literature is moving faster than any individual practitioner can track.** Edge VLM papers and open-source releases appear weekly. A practitioner who relies only on techniques they personally know will fall behind a practitioner who systematically tracks the literature.

These nine properties are not assumptions — they are observations about the problem. Any solution that ignores any of them will produce worse results than one that respects them. The components of the system, derived in §5, exist to address these properties.

---

## 3. What needs autonomous reasoning vs. what's deterministic

This is the most important cut in the HLD. Earlier drafts of the build plan defined five agents (Research, Experiment, Training/Compression, Deployment, Evaluation, Ranking). Re-examining each in light of §2:

**Genuinely needs autonomous reasoning (an LLM is doing real work):**

- **Reading research papers and extracting structured claims.** Property P9. This is a comprehension and abstraction task; deterministic code cannot do it.
- **Proposing configurations to try next, given the state of the search.** Property P1, P5, P7. This requires reasoning about which regions of the search space are likely productive, given current results.
- **Triaging when something goes wrong.** When a training run NaNs or a deployment fails, classifying the failure and deciding whether it's a fixable config issue or a fundamental problem requires reasoning.
- **Writing human-readable rationales for recommendations.** Property P8. A Pareto-best candidate needs a written explanation for the approving human, not just a row of numbers.

**Deterministic — does not need an LLM at all:**

- **Running an experiment given a config.** Take config, dispatch training/compression/deployment, capture metrics, write to database. This is a job scheduler with logging.
- **Computing the Pareto frontier.** Strict mathematical operation on metric tuples. Doesn't benefit from reasoning.
- **Choosing the export path for a deployment target.** A `DeviceDescriptor` says "this device supports MLX, CoreML, ONNX." A lookup table picks the right export pipeline. No reasoning required.
- **Running an evaluation harness.** Load model, run it on benchmark data, score outputs, write metrics. VLMEvalKit already does this; we wrap it.
- **Schema validation, sandboxing, canary checks, golden-set regression gates.** All deterministic verifiers.

**Could go either way, depending on design choice:**

- **Parsing model outputs for fuzzy evaluation (LLM-as-judge for captioning).** A small LLM can do this; so can deterministic similarity scoring against references. Decision: use LLM-as-judge only where reference-based scoring is known to correlate poorly with human judgment (captioning, open-ended VQA). Otherwise, deterministic.
- **Deciding when the known-techniques toolkit is exhausted.** Could be an LLM looking at the search history; could be a deterministic threshold-monitor. Decision: deterministic threshold-monitor (per §4.2 below), because reproducibility matters more than nuance here.

**Implication:** **most of the work in this system is deterministic.** The LLM-driven autonomous components are doing high-leverage cognitive work (reading papers, reasoning about configurations, explaining decisions) but they are a minority of the system's code and runtime. Earlier drafts overstated the agent count by assigning LLM-powered agent abstractions to deterministic work. The system is better designed as **a small number of LLM-driven agents wrapped around a larger substrate of deterministic services.**

I will use the terms "agent" and "service" precisely from here on:

- **Agent** = a component whose primary work requires LLM reasoning. Has a permissions manifest, a JSON-schema contract for I/O, a structured decision log.
- **Service** = a component whose work is deterministic. Has an API but not a contract-as-LLM-prompt, no decision log (just operational logs).

This distinction matters because agents need fundamentally more infrastructure than services (sandboxing, hallucination guards, decision logs, shadow-mode promotion). Mis-classifying a service as an agent imposes that infrastructure cost for no benefit. Mis-classifying an agent as a service hides places where the system needs guards it doesn't have.

---

## 4. The two operating modes

Per the goal clarification you provided: optimization starts with well-known techniques. The system only escalates to research-driven exploration when the well-known toolkit appears insufficient — and the escalation decision is made by a human, not the system.

This gives the system two distinct operating modes. The architecture must support both, and the transition between them must be explicit.

### 4.1 Mode A: Known-techniques optimization loop

In Mode A, the system explores a defined search space of well-understood techniques. The toolkit is enumerated and finite (though large):

- **Compression:** FP16, INT8 (per-channel weights, dynamic activations), INT4 (weight-only, group-wise at various group sizes), mixed precision.
- **Architecture choices within known families:** vision-token budgets, input resolutions, encoder/decoder swaps within compatible families.
- **Runtime backends:** MLX, CoreML, ONNX Runtime, llama.cpp / GGUF, per `DeviceDescriptor`.
- **Decode strategies:** greedy, top-k, speculative decoding (when a draft model is available).
- **Training-time techniques:** fine-tuning from open weights, caption-only cached distillation from open-license teachers, structured pruning at fixed sparsity levels.

Mode A is essentially a sophisticated *informed sweep* with on-device measurement and Pareto tracking. It does not require literature ingestion or hypothesis generation. It does require:

- Intelligent sampling (don't run obviously-bad combinations).
- Cheap-filter-before-expensive-measurement (compute proxy metrics on Mac before pushing to iPhone/Pi).
- Per-device Pareto frontier maintenance.
- Stateful learning from prior results.

Mode A is the simpler, easier-to-build, faster-to-iterate mode. It produces the **known-techniques baseline** — the ceiling of what can be achieved without any research-driven exploration. This baseline is the anchor against which Mode B's value is later judged.

### 4.2 Mode B: Research-driven exploration

Mode B is invoked when monitored signals suggest the Mode A toolkit has been exhausted *and* the system is still meaningfully short of the success criteria. Mode B brings in literature ingestion, hypothesis generation, and human-assisted implementation of novel techniques.

The trigger for Mode B is **not automatic.** Per your guidance: the system monitors a defined set of signals, and when thresholds cross, it raises a *Decision Dossier* to the human approval queue. The human decides whether to escalate.

The monitored signals for the escalation decision are:

| Signal | What it measures | Threshold meaning |
|---|---|---|
| **Pareto frontier movement velocity** | How many new Pareto-improving points have been found in the last N experiments | Stagnation indicates the known toolkit is running out of headroom |
| **Search space coverage** | What fraction of the defined Mode A search space has been explored | Low coverage + frontier stagnation is suspicious — maybe sampling is the problem, not toolkit exhaustion |
| **Headroom against success criteria** | Gap between current Pareto frontier and target (Liquid AI's benchmark) | A small gap with stagnation is "we're almost there"; a large gap with stagnation is "Mode A may not be enough" |
| **Per-axis progress** | Quality, size, latency, memory considered separately | Tells the human *which kind* of research to focus on — quality stagnation suggests training-side techniques; latency stagnation suggests runtime-side |
| **Cost spent vs. budget** | Wall-clock time, GPU spend, electricity | Bounds the decision: "we've spent 60% of the budget and aren't there yet" is a different decision context than "20% spent" |

When the signals cross thresholds, the system assembles a **Decision Dossier** — a structured artifact containing all five signals, the current Pareto frontier, a list of unexplored regions of the Mode A search space (if any), and the system's recommendation — and posts it to the approval queue. The human reads the dossier and decides:

- **Stay in Mode A** with refined sampling.
- **Escalate to Mode B** to bring in research-driven techniques.
- **Adjust the success criteria** if the goals themselves seem unrealistic.
- **End the project** if the cost/benefit no longer justifies continuation.

This is genuinely different from auto-escalation. The human is the one who decides the system is stuck, based on legible evidence the system has assembled.

### 4.3 Steady-state interaction between modes

Once Mode B has been invoked, the system runs both modes in parallel. Mode A continues exploring the (now possibly extended) known-techniques search space; Mode B feeds new techniques into that search space as they're successfully implemented and validated. The system is no longer "in Mode A" or "in Mode B" — both are running, and the Mode B output expands what Mode A can search over.

A Mode B technique becomes "promoted" once it has been:

1. Successfully implemented (auto-implemented for config-change techniques; human-implemented for code-requiring techniques).
2. Validated on a small canary slice without catastrophic regressions.
3. Run on the full evaluation set with proper statistical methodology (multiple seeds, confidence intervals where appropriate — see §6.4).
4. Approved by the human as eligible for inclusion in the Mode A search space.

After promotion, the technique is just another knob in Mode A. The system has *grown its own toolkit*.

---

## 5. Where humans gate

Per the single principle stated earlier: humans gate **anything irreversible, expensive, or epistemically risky.** Applied to the two operating modes, this produces a concrete list.

### 5.1 Always-gated (require human approval before execution)

| Gate | Reason | Mode |
|---|---|---|
| **Deploy to a real device** | Irreversible (user trust, real-world consequences if the model is bad) | A and B |
| **Large compute spending** (cloud GPU rental, multi-day training runs) | Expensive | A and B |
| **Change to evaluation metrics or eval set composition** | Epistemically risky (changes what "good" means) | A and B |
| **Promote a Mode B technique into the Mode A search space** | Epistemically risky (a flawed promotion contaminates future search) | B |
| **Escalation from Mode A to Mode B** | Strategic decision, requires the Decision Dossier (§4.2) | Mode transition |
| **Approve a Tier 2 hypothesis for implementation** (see §6.2) | Epistemically risky (commits human implementation time to a hypothesis that may not pan out) | B |
| **Cross-phase transitions** (Phase 0 → 1 → 2 → 3) | Project-level strategic decisions | All |

### 5.2 Never-gated (the system acts autonomously)

- Running individual experiments within an already-approved search space.
- Updating the Pareto frontier with new measurements.
- Filtering candidates with cheap proxy metrics.
- Choosing which iPhone/Pi/Mac/runtime to dispatch a given experiment to.
- Reading new papers and generating hypothesis records (Mode B). Posting them to a review queue is autonomous; *approving* them is gated.
- Computing and posting the Decision Dossier when monitored signals cross thresholds.
- Triaging failed runs and retrying transient failures.

### 5.3 Gated-by-thresholds (a designed-in policy, not per-decision approval)

Some decisions warrant pre-approved policies rather than human review of each instance. The human approves the policy once; the system applies it many times.

- **Per-experiment compute spend cap.** Pre-approved budget per experiment; the system kills any run that exceeds it.
- **Canary regression auto-kill.** Pre-approved thresholds (e.g., "if canary loss diverges within 100 steps, kill"); the system enforces them.
- **Sampling-density policy in Mode A.** Pre-approved coverage targets per region of the search space; the system schedules to meet them.

These keep the human out of the decision loop for repetitive judgments while still keeping policy decisions visible.

---

## 6. Proposed component topology

This is what falls out of §§2–5. Some components are agents (LLM-driven), others are services (deterministic). The count is deliberately smaller than the five-agent design in earlier drafts.

### 6.1 The components

**Agents (LLM-driven; require sandboxing, hallucination guards, decision logs):**

1. **Search Strategist Agent.** Maintains beliefs about which regions of the Mode A search space are productive. Proposes which configurations to try next. Triages failures. Writes rationales for the Pareto candidates surfaced for human review. *This is the agent that runs continuously in Mode A.*

2. **Research Analyst Agent.** *(Mode B only.)* Ingests papers and repositories, extracts structured hypothesis records per §6.2 below. Does not generate implementations; does not run experiments. Runs on a slow cadence (weekly, not per-experiment).

**Services (deterministic; standard logging, no LLM-specific infrastructure):**

3. **Experiment Runner.** Takes a config, executes it (training, compression, deployment, evaluation as the config dictates), captures metrics, writes a content-addressed result bundle. Schedules across the two-Mac topology and the iPhone/Pi target devices.

4. **Evaluation Harness.** Wraps VLMEvalKit and our custom photo-memory eval set. Runs on demand, produces structured metric reports. Includes LLM-as-judge for fuzzy captioning evaluation only (everything else is deterministic).

5. **Pareto Tracker.** Maintains per-device Pareto frontiers across all experiments. Computes dominance, identifies frontier movement, exposes the queries that the Search Strategist needs.

6. **Deployment Dispatcher.** Reads a `DeviceDescriptor`, picks the right export pipeline (MLX / CoreML / ONNX / GGUF), packages the artifact, pushes it to the device test harness. Lookup-table logic, no reasoning needed.

7. **Threshold Monitor.** Watches the signals defined in §4.2 (Pareto velocity, search coverage, headroom, per-axis progress, cost). Posts a Decision Dossier to the approval queue when thresholds cross. Deterministic.

8. **Human Approval Queue.** Surface for all gated decisions. Append-only log. Web/CLI interface.

9. **Technique Registry.** The codified Mode A search space, plus all Mode-B-promoted techniques. Each entry has: name, parameter ranges, applicability constraints, implementation reference. The Search Strategist reads from this registry to know what's available.

That's two agents and seven services. Compared to the earlier five-agent design, three "agents" have been demoted to services because their work is deterministic (Deployment, Evaluation, Ranking-as-Pareto-tracking). The earlier Training/Compression Agent has been absorbed into the Experiment Runner because training is a deterministic dispatch operation; the decisions *about* what to train are made by the Search Strategist.

This is a meaningful simplification. The system has the same expressive power as the five-agent design but substantially less infrastructure to build.

### 6.2 The hypothesis record — Research Analyst Agent's output

Per your guidance to "do the best for human to understand and implement easily," hypothesis records are designed as **implementation kits**, not as terse leads. Each record contains:

| Field | Content |
|---|---|
| **Title** | Short technique name |
| **Source citation** | Paper title, authors, venue, year, arXiv ID, GitHub repo if available |
| **Claimed effect** | What the paper says the technique does, in 2–3 sentences |
| **Verbatim excerpts** | The specific passages from the paper that describe the technique. Quoted directly so the human doesn't have to re-read the paper to refresh context. |
| **Original hyperparameters** | The paper's reported configuration |
| **Reported results** | The paper's claimed numbers, including the eval setup that produced them |
| **Applicability check** | Does this work with our model size, our data, our hardware, our quantization? Answered with citations from the paper. |
| **Known failure modes** | Limitations the paper discloses (often in the discussion or limitations section) |
| **Implementation difficulty estimate** | One of: `config-change` (Tier 1, auto-runnable), `minor-code-change` (Tier 2, ~few hours human work), `new-module` (Tier 2, ~day human work), `major-refactor` (Tier 2, escalate to project-level decision) |
| **Proposed codebase insertion point** | Which file/class in our codebase this would extend. A starting point for the human, not final code. |
| **Confidence flags** | Any aspect of the extraction the LLM is unsure about — e.g., "I'm 70% confident the paper means group size 32 in this passage; verify before implementing." |

The confidence flags are particularly important: they make the LLM's uncertainty visible rather than hidden, so the human knows which parts of the record to scrutinize. An LLM-extracted record that pretends to be confident about everything is worse than one that says "I'm not sure about this." This is part of the hallucination mitigation in §6.4.

### 6.3 Two-tier hypothesis handling

**Tier 1 — auto-runnable.** Hypothesis is purely a configuration change to existing techniques. The Search Strategist Agent can add it to the search space without any new code. Examples: "try INT4 with group size 64 instead of 32," "try vision-token budget 96," "use top-k decoding with k=5 instead of greedy."

**Process for Tier 1:** Research Analyst produces the record → applicability check passes → record auto-promoted to a new entry in the Technique Registry → Search Strategist picks it up in the next round of Mode A sampling. Human is not in the loop for Tier 1; the existing Mode A safeguards (canary, golden-set regression, schema validation) catch any failures.

**Tier 2 — code-requiring.** Hypothesis requires new code (a new attention pattern, a new compression scheme, a new vision encoder). Cannot be auto-implemented safely.

**Process for Tier 2:** Research Analyst produces the record → record posted to the Human Approval Queue → human reviews the record, decides whether to invest implementation time → if approved, human implements the technique using the record as their guide (possibly LLM-assisted, but human-reviewed) → human registers the implementation in the Technique Registry → it becomes part of the Mode A search space.

The human is the implementor, not the system. The LLM's job is to make the human's implementation work *as easy as possible*, not to replace it.

> **Amended by §6.5 (ADR-0012, 2026-06-15).** Phase-2 experience refined this: Tier-2 human work should produce a *parameterized capability* the agent then drives (a generic builder + a declarative spec), not a one-off the human hand-builds each time. The "human is the implementor" rule still holds for the irreducible *machinery*; the agent constructs every *instance*. See §6.5 — and note that this is the same seam through which Mode B / Research-Analyst-discovered schemes become agent-applicable.

### 6.4 LLM-driven research failure modes and mitigations

The Research Analyst Agent's failure modes are real and not fully solvable. The HLD addresses them explicitly because they're first-order concerns, not appendix material.

| Failure mode | What it looks like | Mitigation |
|---|---|---|
| **Citation hallucination** | Agent cites a paper that doesn't exist, or attributes a claim to the wrong paper | Every citation has a URL field; a deterministic verifier hits the URL and confirms the paper exists and has the cited title. Records with broken citations are flagged before reaching the human queue. |
| **Misreading the paper** | Agent extracts a claim that the paper does not actually make | The confidence flags from §6.2 surface uncertainty. Verbatim excerpts in the record let the human cross-check the extraction against the source quickly. |
| **Wrong applicability** | Agent says a technique applies when it doesn't (e.g., paper requires a specific architecture we don't use) | The applicability check field is structured: it lists the technique's requirements explicitly and matches against our setup. Mismatches reject the record before human review. |
| **Statistical noise as signal** | Agent (or system) reports an improvement that's within experimental noise | Promotion of a technique from Tier-1-auto-tried to Technique Registry requires multiple seeds and effect size > some threshold. A single lucky run does not promote. |
| **Implementation drift** | Human implements Tier 2 technique, but it doesn't match the paper's description | The verbatim excerpts in the record let the human verify their implementation against the source. The system also runs the technique on a small reproducibility-check task before allowing it into the main search. |

The takeaway: **the LLM is not trusted; it is verified.** Every output of the Research Analyst Agent passes through deterministic checks before reaching a human, and even then is presented with its uncertainty visible.

---

### 6.5 Amendment A — System-Driven Construction (revises §3 and §6.3; per ADR-0012, 2026-06-15)

The original HLD drew the autonomy line at "Tier-2 = code-requiring → the human implements it." Phase-2 experience overruled that boundary as written. After two distillation pilots regressed (P2-D1 caption-only, P2-D2 task-aligned — both distilling *into* the LFM2 benchmark, see ADR-0011), the corrected approach was to **construct** a right-sized student from the Qwen2.5-VL-3B lineage. The decision (ADR-0012, your directive: *"system should do, not human implement"*) was to make model construction a **system capability**, not a human one-off. This section revises the tier model accordingly.

#### 6.5.1 Revised principle

> **Tier-2 human work produces a *parameterized capability*, not a one-off.** A human writes a generic builder **once** (the irreducible machinery — e.g. assembling a multimodal forward pass *is* code). Thereafter the agent constructs every *instance* by proposing a declarative, content-addressed spec, and a deterministic loop builds → trains → evaluates → records it. Construction becomes a search dimension, not a manual task.

The §6.3 rule ("the human is the implementor") still holds — but for the **machinery**, not the **model**. The human's deliverable shifts from "a model" to "a new agent-drivable knob."

#### 6.5.2 The tier model is now three-way

| Tier | What | Who acts | Example |
|---|---|---|---|
| **Tier 1 — config** | Change a parameter of an existing technique | Agent (auto) | INT4 group size 64; vision-token budget 96 |
| **Tier 1.5 — parameterized construction** *(new)* | Propose a content-addressed **spec**; a builder assembles/trains/evals/records it autonomously | **Agent (auto)** | `StudentSpec` → `build_student` → `construction_loop` (ADR-0012, B1.0–B1.3): assemble LM+vision+projector, align, distill, score same-path, write to the ledger |
| **Tier 2 — new machinery** | A technique no existing builder covers | Human writes/extends the generic builder **once** → exposes it as new spec parameters → it *becomes* Tier 1.5 | A new projector architecture, a new compression scheme, a new vision encoder family |

The crucial change: Tier 2 is now a **one-time conversion of "new code" into "new spec parameters,"** after which the agent drives it. We climbed this ladder in practice during B1.0–B1.3: a human wrote the generic `build_student` once (Tier 2), and the Search Strategist now proposes `StudentSpec`s (Tier 1.5) via its `propose_student` tool.

#### 6.5.3 Composition with Mode B / the Research Analyst (the load-bearing part)

This revision is **designed so research-discovered schemes remain applicable** — it must not turn construction into a closed box. A scheme extracted by the Research Analyst (§6.2, Mode B) flows into the system through the **spec schema as the extension seam**:

- **If the scheme fits an existing spec dimension** (a new projector type, vision encoder, distillation objective, quantization scheme, alignment recipe) → it lands as a **new value/parameter** in the spec or Technique Registry → **Tier 1 / 1.5**: the Search Strategist applies it by proposing a spec. No new machinery.
- **If no existing builder covers it** → the human extends a builder **once** (Tier 2) so the scheme becomes new spec parameters → thereafter Tier 1.5.

So the §6.2 hypothesis record's **`implementation difficulty`** and **`proposed codebase insertion point`** fields now point precisely at this seam: the Research Analyst's job becomes *"map this paper's technique onto a spec parameter, or onto a builder extension."* Mode B **grows the space of buildable specs**; the Search Strategist **searches that space**. Construction (ADR-0012) and research-ingestion (Phase 3) thus compose by construction rather than colliding — the same `StudentSpec`/registry that the agent searches today is what tomorrow's literature techniques plug into.

> **Design obligation this creates:** the spec schema and the (still-to-be-built, §8) Technique Registry must be **extensible by adding values/fields**, not by editing core logic. `StudentSpec` (schemas/students.py) is the first instance and should evolve toward the §8 `techniques/`-directory registry vision so that Mode-B promotion = "register a new spec dimension," not "patch the builder." This is the concrete link between Amendment A and the still-open §8 / Research-Analyst work.

#### 6.5.4 Unchanged guards

Everything in §5 (human gates) and §6.4 (verification, not trust) still applies. Every constructed artifact is content-addressed by spec hash, recorded in the experiment ledger, and scored on the same inference path as the benchmark (P2-1.3). **Promoting** a new builder or scheme into the *default* search space remains epistemically gated (§5.1) — the agent may construct and measure freely, but widening the standing toolkit is still a human decision.

---

## 7. Coordination and topology

### 7.1 Logical coordination

The system has a simple control-flow pattern: a job queue connects the Search Strategist Agent (producer of experiment requests) to the Experiment Runner (consumer). Results flow back to the metrics database, which the Pareto Tracker reads. The Threshold Monitor watches the metrics database and posts to the Approval Queue when thresholds cross.

In Mode B, the Research Analyst Agent runs on a separate, slower cadence — weekly, not per-experiment. It writes hypothesis records to a review queue. Approved Tier 1 records become Technique Registry entries; approved Tier 2 records become human implementation tasks.

There is no complex orchestration. The system is fundamentally a job queue with a metrics database and an approval queue. Everything else is either a producer or a consumer of those three.

### 7.2 Physical topology

Two Macs of asymmetric spec (as established earlier):

- **Compute Mac** (Mac mini M4 16 GB initially; M5 Pro 32 GB when available): runs the Experiment Runner, Evaluation Harness, training jobs, reference inference. This is where MPS bandwidth matters. Both machines share the `measurement_and_training` role and use the same measurement harness; `device_id` in each `MetricsReport` distinguishes which machine produced it.
- **Agent Mac** (M4 16 GB): hosts the Search Strategist Agent's local LLM, the Pareto Tracker, the Threshold Monitor, the Approval Queue, the metrics database, the dashboard. This is the human-facing machine.

Plus the target devices: an iPhone (15 Pro or 16 Pro) and a Raspberry Pi 5, both on the same local network, both running thin measurement harnesses that report metrics back to the Compute Mac.

The Research Analyst Agent runs on the Agent Mac when active, but uses a **frontier API** (Claude or GPT-class) for its actual reasoning — not a local model. This is because:

- It runs weekly, not per-experiment. API cost is bounded (order of $10–30/week).
- Local 14-35B models hallucinate citations on dense technical papers — exactly the failure mode this agent must avoid.
- The hallucination mitigations in §6.4 work better with frontier-class reasoning.

The Search Strategist Agent runs on a local model (Qwen3-Coder-7B Q4 on the M4 Agent Mac) because it runs continuously and its work is more bounded (it reasons about a defined search space, not arbitrary literature).

### 7.3 Bridge between Macs

Per earlier discussion: deliberately boring. Phase-appropriate progression:

- Phase 0–1: shared directory via syncthing or NFS. Agent Mac writes job requests to a directory the Compute Mac watches; results come back the same way.
- Phase 2+: upgrade to a FastAPI service on the Compute Mac when the file-watching pattern's edge cases start hurting. Still order-of-magnitude simpler than any distributed framework.

No Ray, no Kubernetes, no Celery. Two long-running Python processes that talk over HTTP. The system's complexity should be in the *agents and decision logic*, not in the plumbing between machines.

### 7.4 Failure modes and resilience

| Failure | Effect | Recovery |
|---|---|---|
| Agent Mac crashes | Search planning pauses; Compute Mac finishes current job and idles | Restart Agent Mac; resume from persisted job queue |
| Compute Mac crashes mid-experiment | Current experiment lost; Agent Mac sees a job timeout | Agent re-queues the job; cost is one wasted experiment |
| Network flake between Macs | Jobs queue up; nothing destroyed | Both sides tolerate disconnection because all comms are via queue, not synchronous RPC |
| iPhone or Pi unreachable during measurement | Experiment fails with `device_unreachable` | Re-queue with backoff; if persistent, raise to human |
| Local LLM on Agent Mac fails | Search Strategist stalls | Restart local LLM; fall back to cheap API if recurring |
| Frontier API unavailable for Research Analyst | Mode B ingestion paused | Mode A continues unaffected; humans can manually queue paper summaries while API is out |

The system tolerates partial failures gracefully because the job-queue model makes everything resumable. There is no in-memory state that, if lost, breaks the system.

---

## 8. The human implementation environment

Per your earlier guidance: if the human is going to implement Tier 2 techniques, the codebase architecture has to make this easy. This shapes the repository layout.

**Required architectural elements:**

- A **`techniques/`** directory in the repository. Each subdirectory is one technique, with a standardized structure: `metadata.yaml` (declaring parameter ranges and applicability), `implementation.py` (the actual code), `tests/` (validation tests), `paper_reference.md` (the paper/repo it came from).
- A **technique registry** as a singleton service that scans `techniques/` at startup and exposes the search space to the Search Strategist. Adding a new technique is just adding a new subdirectory; no central registry edit needed.
- A **code scaffold generator** that, given a Tier 2 hypothesis record, produces the `techniques/<name>/` directory pre-populated with stubbed files and the relevant paper excerpts. The human starts from a structured starting point, not a blank file.
- An **`integration` test framework** that exercises each technique on a small canary task before it's eligible for the main Mode A search. Catches Implementation Drift (per §6.4).
- A **technique audit log** so the lineage of every promoted technique is traceable: who implemented it, when, against which hypothesis record, with what canary results.

This is "the human implementation environment" — the affordances that make Tier 2 work feel like filling in a template rather than starting from scratch. Without these, the project regresses to ad-hoc implementation and loses the system-ness it's after.

---

## 9. Implications for the Goals document

Writing the HLD surfaced several issues with the current Goals document. Listing them here for discussion; not editing Goals yet.

**G1 — §4 (Scope) has solution-shape items.** You already flagged this. The HLD now contains the architectural commitments (multi-agent, the two-Mac topology, the specific agent count); Goals should narrow its scope section to problem-shape items only (what tasks, what devices, what kind of success).

**G2 — Phase numbering and content.** Current Goals doc has:
- Phase 0: foundations and baselines
- Phase 1: single-agent compression loop
- Phase 2: multi-agent split + training
- Phase 3: research ingestion

Under the HLD's Mode A / Mode B framing, this should be:
- Phase 0: foundations and reference baselines (unchanged in intent)
- Phase 1: Mode A loop with the Search Strategist Agent + core services (was "single-agent" before; now reflects the actual minimum viable system)
- Phase 2: Mode A at full capability (extended search space, training pipeline, full set of services, polished approval gates). *No multi-agent split needed because the HLD has only two agents total; the Research Analyst comes in Phase 3 alone.*
- Phase 3: Mode B brought online with the Research Analyst Agent and the Decision Dossier escalation pattern.

This is a meaningful re-shaping. The earlier phasing assumed five agents to be split apart in Phase 2; the HLD's two-agent design makes that split unnecessary.

**G3 — Success criteria need a Mode B item.** Current criteria are about autonomy (P1–P3), quality (S1–S2), and model-level wins (T1–T2). None of them test whether Mode B works. A new criterion is needed: *"The system has ingested at least one technique from recent literature, implemented it (Tier 1 or Tier 2), and validated that it improves the Pareto frontier beyond the Mode A baseline."* Without this, Mode B is built but its central value claim is untested.

**G4 — The Decision Dossier pattern should be a goal-level commitment.** It's the mechanism by which humans gate the most consequential decision (Mode A → Mode B escalation). It deserves a sentence in the Goals doc, not just in the HLD. Specifically: "Strategic decisions (mode escalation, scope changes, project termination) are gated by Decision Dossiers — structured artifacts presenting the system's evidence to the human."

**G5 — The non-goal "smaller is preferred at equal quality" should become a first-class preference in the Pareto formulation.** Currently it's stated as a non-goal in Goals §4. In the HLD it's a tie-breaking rule in the Pareto Tracker. These should be consistent: either it's a real first-class preference (and the Goals doc should state it as such), or it's a default tie-breaker (and Goals §4 doesn't need to mention it as a non-goal).

These five items would be the basis for a Goals v2 revision after you review this HLD.

---

## 10. Open design questions

Things I didn't decide in this HLD, because the right answer depends on inputs I don't have:

**D1 — Confidence flagging in hypothesis records.** §6.2 says the LLM should attach confidence flags. The granularity (per-field vs. overall record) and format (numerical score vs. categorical "low/medium/high") are open. Recommendation: per-field categorical, because numerical scores from LLMs are not calibrated.

**D2 — Statistical methodology for technique promotion.** §6.4 requires multiple seeds and effect-size thresholds before a Mode B technique is promoted to the Mode A search space. The exact methodology (number of seeds, which effect-size measure, what threshold) is a methodological choice that deserves more thought than this HLD gives it. Recommendation: start with 3 seeds and a Cohen's-d threshold of 0.5 for promotion, with the threshold itself being a tunable parameter in the Threshold Monitor's policy.

**D3 — Where does prior-art search end?** The Research Analyst Agent ingests "papers and repositories." How is the corpus defined? arXiv categories? A curated reading list? Human-queued papers? Recommendation: a hybrid — auto-ingest a defined arXiv category subset (cs.CV, cs.LG with edge/efficiency keywords) plus a human-curated paper queue for higher-value targets.

**D4 — How is the Pareto Tracker's "tie" defined for the smaller-is-better preference?** Two models with quality scores within 1% of each other — are they "tied" and broken by size, or is the 1% considered meaningful? Recommendation: configurable per axis, defaulting to "within 1 standard error" as tie definition.

**D5 — Does the Search Strategist Agent's belief state persist across phases, or reset at each phase?** Persistence makes the system learn over time. Reset makes each phase's claims independent and clean. Recommendation: persist within a phase; reset at phase transitions for clean validation. Document the reset in the experiment ledger so it's traceable.

These should be discussed and locked before the Detailed Plan.

---

## 11. Summary of the HLD's differences from earlier drafts

For the reader comparing this against the earlier build-plan content:

| Topic | Earlier draft | This HLD |
|---|---|---|
| Number of "agents" | 5 (Research, Experiment, Training/Compression, Deployment, Evaluation, Ranking) | 2 agents (Search Strategist, Research Analyst) + 7 services. Same expressive power, much less infrastructure. |
| Research integration | Phase 3 add-on | Mode B, with explicit Decision-Dossier-gated escalation from Mode A |
| Phase 2 content | "Multi-agent split + training" | Mode A at full capability. No "split" is needed because the system was never one monolithic agent in the first place. |
| Mode of LLM use for research | Implicit "Research Agent generates hypotheses" | Explicit: LLM-as-analyst, hypothesis records as implementation kits, Tier 1 / Tier 2 split, verbatim-excerpt-backed records, confidence flags |
| Human-in-the-loop | Listed without principle | Derived from single principle (irreversible / expensive / epistemically risky) and applied systematically |
| Escalation trigger | Implicit | Explicit: Threshold Monitor + Decision Dossier + human decision |
| Tier-2 (code-requiring) work | Human implements each technique/model | **Amendment A (§6.5):** human writes a generic builder *once*; agent drives every instance via a declarative spec (Tier 1.5). Research-discovered schemes plug in as new spec parameters. |

The system is meaningfully simpler than earlier drafts, and the parts that remain are better justified.

---

*Next document: "Multi-Agent VLM Optimization System — Detailed Plan," to be written after the Goals doc is revised per §9 and the open design questions in §10 are addressed.*
