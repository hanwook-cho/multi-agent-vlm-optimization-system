# Multi-Agent System for VLM Optimization — Goals (v3)

*Goal hierarchy for the project, from the ultimate aim down to per-phase objectives and deliverables. Revision v3 sharpens the central claim from "produce a competitive model" to "compress the time required to produce a competitive edge VLM from team-months to solo-months" — and shifts the Phase 1-2 starting point from already-optimized LFM2-VL-450M to a general-purpose Qwen2.5-VL-3B so the optimization journey being demonstrated is meaningful.*

**Audience.** Project owner (solo developer), future collaborators, external reviewers.
**Status.** Draft v3.
**Last updated.** May 11, 2026.
**Companion documents.** "VLM_Optimization_HLD.md" (architecture), "VLM_Optimization_PriorArt.md" (prior art survey), "VLM_Optimization_DetailedPlan_Phase0.md" (Phase 0 execution plan; subsequent phase plans to be written).

---

## 1. The ultimate goal

Build a closed-loop, autonomous agent system that compresses the time required to produce a competitive edge vision-language model — from the team-months of focused expert work that produced models like LFM2-VL-450M, SmolVLM-500M, and MiniCPM-V, to solo-developer-months using the system as the optimization tool.

The system starts from well-known optimization techniques (Mode A) and escalates to research-driven exploration (Mode B) when known techniques are exhausted. It autonomously navigates the full optimization journey from a general-purpose, non-edge-suitable VLM down to a competitive edge model, with human gates only on consequential decisions.

**The system is the deliverable. The compressed time-to-result is the central claim. A competitive model is the proof-of-work.**

The system is judged successful if it independently produces a vision-language model that:
- Is competitive with the leading published edge VLMs (LFM2-VL-450M, SmolVLM-500M, MiniCPM-V 4.6, FastVLM-0.5B) on shared benchmarks, at comparable or smaller footprint
- Runs on iPhone (16 Pro or later). *(Raspberry Pi 5 was originally a co-equal target but is **deferred** — current scope is iPhone/Apple-Silicon; iPhone-only is sufficient for the core time-compression claim, see README and HLD §7.2. Pi 5 becomes Phase-3 debt if hardware is unavailable.)*
- Was produced in solo-developer-months of calendar time, versus the multi-month expert team efforts that produced the reference models

The autonomous-system claim and the time-compression claim are what make this project a contribution. The model is the evidence both claims hold.

## 2. Why this matters, and the position relative to prior art

Edge VLM deployment today requires manual orchestration across many decisions: model architecture, quantization scheme, runtime backend, input resolution, vision-token budget, per-device export path. The choices interact non-obviously, and they must be re-made for every new task and every new device. Skilled practitioners take substantial expert team effort to produce one good model in this space: Apple's FastVLM is the result of years of FastViT/FastViTHD encoder research; Liquid AI's LFM2-VL family represents over a year of focused iteration; SmolVLM came out of a multi-month systematic research effort published as a paper. The premise of this project is that *most of this orchestration can be automated, compressing the calendar time from team-months to solo-months* without sacrificing the quality of the result.

This is a claim about the *method*, not about any single model. A method that compresses optimization time generalizes; a single model is just a model.

The prior-art survey (companion document) identified four lineages of related work:

- **Hardware-aware AutoML / NAS** is mature (AMC, EfficientNet-EdgeTPU, BatchQuant, FOX-NAS) but pre-LLM and not VLM-targeted.
- **LLM-driven AutoML** is recent (MONAQ, Trirat et al., AutoMaAS) and most directly informs this project's Search Strategist Agent, but no existing system targets VLMs, uses real-device measurement consistently, or separates known-techniques exploitation from research-driven exploration.
- **AI-Scientist (Sakana AI)** is the closest in agent topology — published in *Nature* in March 2026 — but aims at producing research papers rather than deployable models. Its documented failure modes (hallucinated citations, statistical-noise-as-signal, structural errors per Beel, Kan & Baumgart, Feb 2025) are exactly what this project's design choices specifically mitigate.
- **Production edge inference frameworks** (LFM2-VL, FastVLM, SmolVLM, MiniCPM-V, MLX, llama.cpp) are the outputs of human-driven optimization. Matching their performance autonomously, in a fraction of the calendar time, is the success criterion. Beating them outright is a stretch goal.

This project occupies a specific point in this landscape: **bounded autonomy with verifiable outputs for ML optimization, in compressed time**, distinct from AI-Scientist's "fully autonomous research" (which produces papers with documented quality problems) and from existing AutoML's "search without literature ingestion" (which can't extend its own toolkit). The Mode A / Mode B escalation pattern, the LLM-as-analyst hypothesis-record design, and the *explicit time-compression claim* together distinguish this project from prior art.

## 3. Success criteria

Success is judged at the end of Phase 2 (core claim) and Phase 4 (generalization claim). Phase 3 is complementary but not part of the core success criterion.

**Primary criteria — autonomy:**

| # | Criterion |
|---|---|
| P1 | The system produces a deployable model without manual configuration tweaking. The human approves deploys but does not edit configs to make experiments succeed. |
| P2 | The system's output includes a per-device Pareto frontier with at least three candidates per device and a written rationale for the chosen winner. |
| P3 | The chosen model runs on both iPhone (16 Pro or later) and Raspberry Pi 5 (4 GB) with reproducible metrics, with all artifacts (config, weights, build) versioned and re-runnable. |

**Secondary criteria — quality of what the system produces:**

The autonomously-produced model is competitive with at least two of the four published reference models (LFM2-VL-450M, SmolVLM-500M, MiniCPM-V 4.6, FastVLM-0.5B) on the shared benchmark subset (RealWorldQA, MMBench dev-en, POPE), measured via VLMEvalKit on identical slices. "Competitive" is defined as:

| # | Criterion |
|---|---|
| S1 | Within 10% of the reference model's quality scores on at least two of the four references (more stringent at ≤ 5% is stretch). |
| S2 | On-disk size ≤ the chosen reference's, at matched quantization (Q4_0 vs. Q4_0). |
| S3 | Starting point of the optimization is documented and was *not* an already-edge-optimized model. (The Phase 2 starting point is Qwen2.5-VL-3B; see §5.) |

**Temporal criterion — the time-compression claim:**

This is what makes the project meaningful beyond "produced one more model."

| # | Criterion |
|---|---|
| T1 | The calendar time from project start (Phase 0 Week 1) to Phase 2 exit is documented to within ±1 week. |
| T2 | A best-effort comparison is documented showing the calendar time for the reference models' development cycles (using public release histories: SmolVLM v1→v2, LFM2-VL → LFM2-VL, FastViT → FastViTHD → FastVLM). |
| T3 | Project calendar time to Phase 2 exit is meaningfully shorter than the median reference timeline. The honest baseline target: solo developer in 4-6 calendar months versus reference teams' 6-18 calendar months. |

**Generalization criterion — judged at Phase 4:**

The time-compression claim is meaningfully stronger if it generalizes. Phase 4 (now in project scope, see §5) repoints the system at a second task and produces a competitive model in similar time.

| # | Criterion |
|---|---|
| G1 | Phase 4 reaches a competitive model on a second task in ≤ 1 month of solo calendar time. |
| G2 | Only YAML configs (objective registry, device library, eval set hashes) changed — no agent code rewrites. |

**Stretch criteria — model-level wins beyond validation:**

| # | Criterion |
|---|---|
| X1 | The autonomously-produced model strictly improves over at least one reference on at least one metric (quality, size, latency, memory) on at least one device. |
| X2 | Raspberry Pi 5 inference at ≥ 2 tokens/sec for the chosen model. (FastVLM does not run usefully on Pi 5 at all; matching or beating LFM2-VL-450M's Pi performance is the bar.) |

**Ordering and what "success" means:**

- The project is **fully successful** if P1-P3, S1-S3, and T1-T3 are all met. The system worked, produced a competitive model, in compressed calendar time.
- The project is **strongly successful** if the above plus G1-G2 are met — the time-compression is shown to generalize across tasks.
- The project **succeeds at its core claim** if P1-P3 and T1-T3 are met but S1-S3 are only partially met (e.g., quality is 15% behind references rather than 10%). The framing in that case: *"the system compressed calendar time substantially, with some quality cost; further iteration would close the gap."*
- The project **fails its core claim** if P1 is not met (system didn't act autonomously) or T3 is not met (no meaningful time compression). Quality alone (S1) without autonomy or time compression doesn't validate the project.

**This ordering is important.** The system's value is autonomy + time compression. Quality is the evidence those claims hold. A model that beats references but required hand-tuning to get there is not a system success.

## 4. Problem boundary

The project's problem boundary is intentionally narrow so that success is measurable and the timeline is solo-developer-tractable.

**Tasks addressed:** Photo-memory VLM as the primary task — captioning, photo VQA, grounded description, and embeddings as a side-output for retrieval. A second task is added in Phase 4 to validate generalization (specific second task to be chosen at Phase 4 kickoff based on what the project has learned by then; candidates include on-device OCR, grounded scene understanding for AR, and small-document understanding).

**Devices addressed:** iPhone 16 Pro (or later) and Raspberry Pi 5 (4 GB). Other devices (Android, Jetson, Pi 4, Mac as a deployment target) are not in scope; the DeviceDescriptor design (per HLD §2.4) preserves the option to add them later, but adding them is not a project goal.

**Mode of optimization addressed:** Mode A (known-techniques optimization loop) in Phase 1-2. Mode B (research-driven exploration) is brought online in Phase 3 — both to demonstrate Mode B's value and to apply it to LFM2-VL-450M (the "squeeze more from an already-optimized model" complementary claim, see §5).

**Style of success:** Reproducible, measurable, on real hardware. No synthetic benchmarks, no LLM-predicted-as-stand-in-for-measurement, no "trust me it would work."

**Non-goals (not "out of scope" — actively chosen non-goals):**

- Maximum autonomy at any cost. Humans retain approval authority on architecture changes, evaluation metric changes, mode escalation, and device deploys.
- Maximum model quality at any cost. Smaller is preferred at equal quality. When two configurations tie on quality, the Pareto Tracker picks the smaller one.
- Building one specific best model. If the system produces three recommended models for three slightly different objectives, that is a feature.
- Production-ready developer tooling. The system validates the approach; productization is future work, decided after Phase 2 results.

## 5. Phase structure

The project runs as five phases (counting Phase 0). Phase 0-2 establish the core claim (system + time compression). Phase 3 demonstrates the research-driven exploration capability and applies it to an already-optimized model. Phase 4 demonstrates generalization to a second task. Each phase's deliverables include both **technical artifacts** (code, models, results) and **public artifacts** (repository milestones, blog posts). Public artifacts serve the developing-in-public conduct rule (§6) and the path-keeping-open strategy (§7).

Public release of the repository is at end of Phase 1, not Day 1 — see §6.

### Phase 0 — Foundations & reference baselines

**Goal:** Stand up the project infrastructure and lock in measured reference baselines for the four small-edge VLMs (LFM2-VL-450M, SmolVLM-500M, MiniCPM-V 4.6, FastVLM-0.5B) on the target devices, plus measure the Phase 1-2 starting point (Qwen2.5-VL-3B) on Mac as the unoptimized "before" picture.

**Why this phase exists:** Before any optimization claim can be made, we need rigorously measured reference numbers on the actual hardware. Vendor benchmarks use different harnesses, different precision, sometimes different prompts. Phase 0 is the anchor. Adding Qwen2.5-VL-3B as a baseline (Mac-only — it won't run on Pi 5 4 GB, confirming the need for size reduction) makes the Phase 2 "from 3B general to 450M-class edge" claim concretely measurable.

**Exit criteria:**

| # | Criterion |
|---|---|
| 0.1 | LFM2-VL-450M runs on iPhone 16 Pro (via Liquid's LEAP SDK / Apollo app) and Pi 5 4 GB (via GGUF + llama.cpp). Metrics logged. |
| 0.2 | FastVLM-0.5B runs on iPhone 16 Pro via apple/ml-fastvlm demo. Metrics logged. Pi 5 non-viability documented. |
| 0.3 | SmolVLM-500M and MiniCPM-V 4.6 run on iPhone 16 Pro and Pi 5 4 GB. Metrics logged. (Where one doesn't fit on Pi, that's documented.) |
| 0.4 | Qwen2.5-VL-3B runs on the Mac mini (M4, 16 GB). Metrics logged. Non-fit on Pi 5 4 GB documented as "expected and confirmed." M5 Pro 32 GB measurement logged when that machine becomes available (does not block Phase 0 completion). |
| 0.5 | A frozen public-photo evaluation set exists: ≥ 200 photos from Flickr30k/COCO/Open Images, ≥ 100 captions, ≥ 100 VQA pairs, hash-pinned. |
| 0.6 | Reference models evaluated against the eval set and a small slice of RealWorldQA, MMBench dev-en, POPE via VLMEvalKit. |
| 0.7 | Repository contains JSON schemas for all agent/service contracts (per HLD §6), plus `ExperimentConfig`, `MetricsReport`, `AgentDecision`, `DeviceDescriptor`, `HypothesisRecord`. |
| 0.8 | A dashboard view renders reference Pareto markers on per-device frontier plots, including the Qwen2.5-VL-3B starting point clearly marked as "not edge-viable." |
| 0.9 | Spike evaluation completed: build-vs-adopt decisions made for literature-ingestion tools (PaperQA2, OpenScholar, AI-Scientist v2). Decision documented. |
| 0.10 | `THIRD_PARTY.md` documents license posture for all third-party material. |

**Phase 0 deliverables:**

| # | Deliverable | Type |
|---|---|---|
| D0.1 | Private GitHub repository (Apache 2.0 licensed; public at end of Phase 1) with README explaining the project. | Infrastructure |
| D0.2 | The HLD, Goals (this doc), and Prior Art documents committed to `docs/`. | Documentation |
| D0.3 | A reproducible measurement script. | Technical artifact |
| D0.4 | The frozen evaluation set, with redistribution-safe manifest. | Technical artifact |
| D0.5 | JSON schemas for all contracts, formally validated and version-tagged. | Technical artifact |
| D0.6 | Draft blog post (publishes with Phase 1 reveal): "Designing an autonomous VLM optimization system, part 1: prior art and what's still open." | Public artifact (deferred) |

**Duration estimate:** 4-5 weeks solo full-time-equivalent. The added Qwen2.5-VL-3B and SmolVLM/MiniCPM-V baselines add ~1 week to the original Phase 0 estimate.

### Phase 1 — Mode A loop on small-model compression

**Goal:** Implement the Search Strategist Agent plus core services (Experiment Runner, Pareto Tracker, Evaluation Harness, Deployment Dispatcher) to drive an autonomous Mode A optimization loop on a small-model starting point. This phase validates the loop works before Phase 2 attempts the harder 3B → 450M optimization.

**Why this phase exists:** Phase 2 is ambitious (general 3B → competitive edge model). Before tackling it, prove the closed-loop optimizer works end-to-end on a tractable starting point. Phase 1 uses LFM2-VL-450M as starting point and runs compression/runtime sweeps — same shape as Phase 2 but with much smaller compute requirements. If Phase 1 cannot produce a Pareto improvement here, attempting Phase 2 is wasteful.

**Exit criteria:**

| # | Criterion |
|---|---|
| 1.1 | The Search Strategist Agent (one process, conforming to HLD schemas) reads experiment configs, dispatches them to the runner, collects metrics, and updates the Pareto frontier autonomously. |
| 1.2 | Autonomous sweep over (quantization × vision-token budget × resolution × runtime backend) executed on iPhone 16 Pro and Pi 5 4 GB, starting from LFM2-VL-450M. |
| 1.3 | At least one configuration on the Pareto frontier improves on baseline LFM2-VL-450M Q4_0 in size, memory, or latency on at least one device, with quality regression ≤ 5% on the eval set. |
| 1.4 | The improvement was achieved without manual configuration edits after initial setup. Human approval was required only for the final deploy candidate. |
| 1.5 | The Decision Dossier scaffold is in place (Threshold Monitor exists, signals defined, dossier format implemented) even though Phase 1 doesn't need it to fire. |
| 1.6 | Repository goes public at Phase 1 exit, with the Phase 0+1 blog post as the reveal artifact. |

**Phase 1 deliverables:**

| # | Deliverable | Type |
|---|---|---|
| D1.1 | Working Search Strategist Agent code in the now-public repository. | Public artifact |
| D1.2 | Phase 1 Pareto frontier plot with baseline and improved point marked. | Public artifact |
| D1.3 | The Phase 0+1 combined reveal blog post (~2500 words): "Building an autonomous VLM optimization system: prior art, the design, and Mode A working end-to-end." | Public artifact |
| D1.4 | A reproducible experiment bundle demonstrating the Phase 1 improvement. | Technical artifact |
| D1.5 | Documentation of the Search Strategist's reasoning policy. | Public artifact |

**Duration estimate:** 4-5 weeks solo.

### Phase 2 — Full system + general-to-edge demonstration *(the core claim)*

**Goal:** With the full system online (all services, training pipeline, distillation, approval queue, dashboard), autonomously produce a competitive 450M-class edge model **starting from Qwen2.5-VL-3B** — a general-purpose, not-edge-optimized VLM. Document the calendar time taken.

**Why this phase exists:** This is the phase that validates the project's central claim. *The system autonomously navigates the full optimization journey from a general-purpose 3B VLM down to a competitive edge model on both target devices, in solo-developer calendar time, without manual config tweaking.* The starting point matters: Phase 1's "compress LFM2-VL-450M further" doesn't prove the system does the hard optimization work, only that it polishes. Phase 2's "Qwen2.5-VL-3B → competitive 450M" is the hard optimization work itself.

**Exit criteria:**

| # | Criterion |
|---|---|
| 2.1 | All Phase 1 components plus: training/fine-tuning/distillation pipeline, Deployment Dispatcher with full DeviceDescriptor support, human Approval Queue, dashboard with all five reference markers (including Qwen2.5-VL-3B as the starting point) and all candidate models. |
| 2.2 | Distillation pipeline operational: the teacher (**Qwen2.5-VL-3B** — the Phase-2 pivot, ADR-0011; *caption-only* distillation was an early variant that regressed) generates a **task-aligned** target cache (grounded Q&A / MCQ) once; student training reuses the cache. Justifies time/compute. |
| 2.3 | Starting from Qwen2.5-VL-3B, the system produced a 450M-class deployable model without manual configuration tweaks after setup. (P1.) |
| 2.4 | The system's output includes a per-device Pareto frontier with ≥ 3 candidates per device and a written rationale. (P2.) |
| 2.5 | The chosen model runs on iPhone 16 Pro and Pi 5 4 GB with reproducible metrics. (P3.) |
| 2.6 | The chosen model is within 10% of at least two of the four reference small-edge VLMs on the benchmark subset, at ≤ the chosen reference's on-disk size at matched quantization. (S1, S2.) |
| 2.7 | Calendar time from project start (Phase 0 Week 1) to Phase 2 exit is documented. Reference-model development cycles are documented in parallel. The time-compression delta is the headline number. (T1-T3.) |

**Phase 2 deliverables:**

| # | Deliverable | Type |
|---|---|---|
| D2.1 | A working end-to-end system, documented well enough for reproduction. | Public artifact |
| D2.2 | The Phase 2 model: weights, configs, build artifacts for iPhone and Pi 5, derived from Qwen2.5-VL-3B, released under inherited license (Qwen license). | Public artifact |
| D2.3 | A blog post (~3500 words): "Phase 2: the autonomous system produced a competitive edge VLM from a 3B general model. Here's the model, the numbers, the calendar time, the cost — and what's left to do." | Public artifact |
| D2.4 | A timeline-comparison chart: this project vs. SmolVLM, LFM2-VL, MiniCPM-V, FastVLM development histories. | Public artifact |
| D2.5 | An arXiv preprint covering the system, results, and time-compression analysis. **Now recommended, not optional** — the time-compression claim is much stronger with a citable preprint. | Public artifact |
| D2.6 | A short demo video. | Public artifact |
| D2.7 | The full experiment log published. | Technical artifact |

**Duration estimate:** 7-9 weeks solo. This is the largest phase. Includes the cached teacher distillation pass (1-3 days of overnight Mac compute, longer if Qwen2.5-VL-3B is heavier than expected; consider $300-600 cloud GPU rental for this step).

### Phase 3 — Mode B research-driven exploration + LFM2-VL-450M squeezing

**Goal:** Bring the Research Analyst Agent fully online. Demonstrate two complementary claims: (a) the system can ingest recent literature and apply research-derived techniques to improve a Pareto frontier, and (b) even an already-expertly-optimized model (LFM2-VL-450M) still has squeezable value when run through the system with literature-driven techniques.

**Why this phase exists:** Phase 2 demonstrates the system optimizes effectively starting from an unoptimized model. Phase 3 demonstrates two further things: that the system can extend its own toolkit through research ingestion (the Mode B capability that distinguishes the project from a sophisticated AutoML sweep), and that the system adds value even on top of expert manual optimization (the "useful as a tool for working teams" complementary claim).

**Exit criteria:**

| # | Criterion |
|---|---|
| 3.1 | The Research Analyst Agent ingests ≥ 10 recent papers/repositories and emits hypothesis records conforming to HLD §6.2's implementation-kit format. |
| 3.2 | Auto-validation rejects records with broken citations or failed applicability checks (rejection rate is non-zero, proving validation works). |
| 3.3 | At least one Tier 1 hypothesis (config-change) auto-runs and is measured against the existing Pareto frontier. |
| 3.4 | At least one Tier 2 hypothesis (code-requiring) is human-implemented from the hypothesis record, run by the system, and measured. |
| 3.5 | At least one literature-derived technique improves the Phase 2 model's Pareto frontier by a statistically meaningful margin (multiple seeds, effect size > defined threshold). |
| 3.6 | The Decision Dossier fires at least once in realistic conditions, presents the human with the signals from HLD §4.2, and the human's decision is logged. |
| 3.7 | Starting from LFM2-VL-450M as a separate experiment track, the system with Mode B enabled identifies and applies at least one literature-derived technique that improves the LFM2-VL-450M Pareto frontier on at least one metric. **This is the "squeeze" demonstration.** |

**Phase 3 deliverables:**

| # | Deliverable | Type |
|---|---|---|
| D3.1 | Working Research Analyst Agent code. | Public artifact |
| D3.2 | A library of hypothesis records produced during Phase 3, demonstrating the format. | Public artifact |
| D3.3 | A blog post (~3000 words): "Phase 3: can an LLM agent read the literature and improve a model? Including one that experts already optimized?" Honest accounting of hits and misses. | Public artifact |
| D3.4 | An updated arXiv preprint (v2) incorporating Mode B results and the LFM2-VL-450M squeezing result. | Public artifact |
| D3.5 | A short documented Mode B technique catalog: which papers were ingested, which techniques surfaced, which made it through validation, which improved the Pareto frontier. | Public artifact |

**Duration estimate:** 5-6 weeks solo. The LFM2-VL-450M squeezing experiment adds ~1 week to the original Phase 3 estimate.

### Phase 4 — Reusability proof on a second task *(now in project scope)*

**Goal:** Repoint the system at a second VLM task (specific task chosen at Phase 4 kickoff). Produce a competitive model for the new task in ≤ 1 month of solo calendar time. Change only YAML configs (objective registry, device library, eval set hashes) — no agent code rewrites.

**Why this phase exists:** Phase 2 demonstrates the system can produce one competitive model in compressed calendar time. That could be luck or a special case. Phase 4 tests whether the time-compression generalizes — the property that makes the project's central claim a *method*, not a *fluke*. Without Phase 4, the time-compression claim is anecdotal. With Phase 4, it's evidence.

**Candidate second tasks (chosen at Phase 4 kickoff based on what the project has learned):**
- On-device OCR (different vision/language balance from photo memory)
- Grounded scene description for AR (heavy on grounding/RefCOCO-style outputs)
- Document understanding for small documents (heavy on layout/text)
- Visual reasoning for charts (small chart QA, like a tiny ChartQA-focused model)

**Exit criteria:**

| # | Criterion |
|---|---|
| 4.1 | A second objective registry exists for the chosen second task, with a frozen eval set hash-pinned. |
| 4.2 | The system runs end-to-end on the second task with only YAML configuration changes — no Python code added or modified beyond what existed at Phase 3 exit. |
| 4.3 | The system produces a competitive model for the second task. "Competitive" defined by relevant published benchmarks for the task (e.g., for OCR, OCRBench or similar). |
| 4.4 | Calendar time from Phase 4 start to Phase 4 exit is documented and is ≤ 5 weeks. (G1.) |
| 4.5 | A retrospective documents what generalized cleanly and what required friction (configs that needed updating, schema fields that turned out task-specific). |

**Phase 4 deliverables:**

| # | Deliverable | Type |
|---|---|---|
| D4.1 | The second model with weights, configs, builds for iPhone and Pi 5. | Public artifact |
| D4.2 | A blog post (~2000 words): "Phase 4: repointing the system at a second task. What generalized, what didn't." | Public artifact |
| D4.3 | Updated objective registry library demonstrating multi-task structure. | Technical artifact |
| D4.4 | Final arXiv preprint (v3) incorporating Phase 4's generalization result. | Public artifact |
| D4.5 | Final retrospective document covering the whole project. | Public artifact |

**Duration estimate:** 4-5 weeks solo.

### Total project timeline estimate (solo developer, full-time-equivalent)

| Phase | Duration | Cumulative | Notes |
|---|---|---|---|
| Phase 0 | 4-5 weeks | 5 weeks | Foundations + 5 reference baselines |
| Phase 1 | 4-5 weeks | 10 weeks | Mode A loop on LFM2-VL-450M; repo goes public at end |
| Phase 2 | 7-9 weeks | 19 weeks | Full system + Qwen2.5-VL-3B → edge model (the core claim) |
| Phase 3 | 5-6 weeks | 25 weeks | Mode B + LFM2-VL-450M squeeze |
| Phase 4 | 4-5 weeks | 30 weeks | Reusability proof on second task |

**Total: ~6-7 months solo full-time-equivalent for the full project (Phases 0-4).**
**Core project (Phases 0-2): ~4-5 months solo full-time-equivalent.**

If working at half capacity (other commitments), double the timeline.

The honest project commitment: if calendar time to Phase 2 exit exceeds ~5 months solo full-time-equivalent, the time-compression claim weakens substantially and the project's core value proposition needs reframing.

## 6. Project conduct: developing in public

The project is conducted as an open-source, developing-in-public effort — but with a deliberate timing: **the repository is private through Phase 0 and Phase 1, and goes public at the end of Phase 1.** This gives the project a cleaner first public impression (a working Mode A loop, not a Week 1 skeleton) and reduces audience pressure during early development. See ADR-0007.

**Repository conduct:**

- **Phase 0-1:** Private GitHub repository, Apache 2.0 license. Commit as if it were public — clean commits, no sensitive data, decisions documented in ADRs — so the public reveal at end of Phase 1 is just a visibility flip, not a clean-up.
- **End of Phase 1:** Repository goes public. The Phase 0+1 combined reveal blog post is the announcement.
- **Phase 2+:** Public throughout. README stays current.
- All architectural documents (HLD, Goals, Prior Art, Detailed Plan, retrospectives) live in `docs/` and are versioned alongside code.
- Decision logs for non-trivial design choices live in `docs/decisions/` (one file per decision, ADR-style).
- Experiment results published as content-addressed bundles. The entire experiment log is part of the public artifact once the repo is public.

**Writing cadence:**

- One technical blog post per phase exit, starting with the combined Phase 0+1 reveal post. Four phase blog posts total (Phase 0+1 combined, Phase 2, Phase 3, Phase 4).
- An arXiv preprint at end of Phase 2 is now **recommended** (was optional) — the time-compression claim is much stronger with a citable preprint. v2 of the preprint at end of Phase 3 incorporating Mode B results. v3 at end of Phase 4 incorporating generalization results.
- Posts are honest about what worked and what didn't.

**Engagement rules:**

- Issues and PRs welcome once public; respond within reasonable time.
- No premature commercialization announcements. The project speaks for itself through working code and honest writing.
- No co-founder discussions during Phase 0-1 (no audience to discover the project yet). During Phase 2-4, opportunistic conversations are fine but no commitments before Phase 2 success.

**What this conduct earns:**

- **Path 1 (portfolio):** The repository and blog posts are the portfolio artifact, verifiable by anyone after Phase 1 public release.
- **Path 2 (co-founder discovery):** Visibility from Phase 1 onward makes inbound interest possible.
- **Path 4 (learning):** Writing publicly forces tighter thinking than writing privately.
- **Path 6 (open-source community):** Direct; community can engage from Phase 1 onward.

## 7. Path-preservation strategy

The four paths the project is keeping open (portfolio, co-founder, learning, open-source community) are largely served by the same actions: build well, write honestly, release publicly at the right moment, don't commit prematurely. The Detailed Plan operationalizes this; the Goals document makes the strategic commitment.

The single most important constraint: **no commercialization decisions before end of Phase 2.** Any earlier decision is made on insufficient evidence. Phase 0-2 produces a working system and a measured competitive model; *that* is the evidence base for deciding what to do commercially, not speculation.

The single most important enabling action: **the project must be visible enough — at the right moment — that the world can find it.** End-of-Phase-1 reveal is the entry point. If the public release and the Phase 0+1 blog post don't happen, Paths 2 and 6 are theoretical only.

## 8. Stakeholders and approval gates

| Role | Held by | Authority |
|---|---|---|
| Project owner | Solo developer (you) | All architecture decisions, all phase transitions, all commercial decisions if applicable. |
| Human approver (in the loop) | Same person — you wear two hats | Approves agent-proposed architecture changes, new eval metrics, mode escalation, device deploys. |
| External readers | Public via repository and blog | Read access; feedback via issues. Not gating authority, but useful signal. |
| Potential collaborators | None at project start; may emerge | Path 2 outcome, not a project structure. |

Phase transitions require all exit criteria for the phase to be met. The solo developer signs off internally. External readers' feedback may inform whether to proceed to the next phase or revise the current one, but does not gate transitions.

## 9. Open questions

Decisions now locked (previously open, now resolved):

| Decision | Resolution |
|---|---|
| iPhone target | iPhone 16 Pro or later. Documented in DeviceDescriptor. |
| Pi 5 RAM tier | 4 GB. Tighter than 8 GB; constrains Mode B candidates. |
| Public repo timing | Private through Phase 0-1; public at end of Phase 1 — **executed** (repo is now public). *(Planned ADR-0008 was never written; decision is recorded here and in the Phase-0 plan §5.5.)* |
| Phase 4 inclusion | In project scope (was future work). Required to validate time-compression generalization. |
| Phase 1-2 starting point | Qwen2.5-VL-3B (general, not edge-optimized). LFM2-VL-450M used as Phase 1 sanity baseline and as Phase 3 squeeze target. |
| License posture | Apache 2.0 for code; LFM Open License inheritance for LFM-derived models; Qwen license inheritance for Qwen2.5-VL-3B-derived models; FastVLM is benchmark-only, no derivative work. *(Planned ADR-0007 was never written; the authoritative record is [`THIRD_PARTY.md`](THIRD_PARTY.md).)* |
| arXiv preprint | Recommended at end of Phase 2; updated versions at end of Phase 3 and Phase 4. The time-compression claim needs a citable artifact. |

Still genuinely open:

**Q1. Stage B target data ETA.** Currently no target dataset for personal photos exists. Phase 0-4 can run entirely on public Stage A data. *Suggested resolution: Plan as if Stage B may never arrive; revisit if it materializes during Phase 3 or later.*

**Q2. Frontier API access for the Research Analyst Agent in Phase 3.** Phase 3's autonomy claim depends on it. *Suggested resolution: assume frontier API access (Claude or GPT-class) is available for Phase 3, budget ~$10-30/week. Document the local-fallback path in the HLD.*

**Q3. Cloud GPU budget for Phase 2.** The Qwen2.5-VL-3B → 450M distillation pass benefits substantially from GPU. *Suggested resolution: budget ~$300-600 for one-time A100 rental during Phase 2 kickoff; decide at Phase 1 exit based on Mac-only feasibility evidence.*

**Q4. Phase 4 second task selection.** Decided at Phase 4 kickoff, not now. Candidates: OCR, grounded scene description, document understanding, chart QA. *Suggested resolution: defer; the right choice depends on what the project has learned and what's most credibly different from photo-memory.*

**Q5. Whether to build or adopt literature-ingestion tools.** Phase 0 spike (exit criterion 0.9) decides this. *Suggested resolution: complete the spike; the decision is locked there.*

---

*Companion documents: "VLM_Optimization_HLD.md" (architecture), "VLM_Optimization_PriorArt.md" (prior art survey), "VLM_Optimization_DetailedPlan_Phase0.md" (Phase 0 execution plan). Subsequent phase plans to be written as each phase begins.*
