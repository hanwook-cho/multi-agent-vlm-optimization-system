# Multi-Agent System for VLM Optimization — Prior Art Analysis

*A survey of existing research and tools that overlap with this project's goals, and an honest assessment of where this project is novel, where it duplicates prior work, and where it might benefit from existing tooling.*

**Status.** Draft v1.
**Last updated.** May 11, 2026.

---

## 1. The headline finding

**There is no existing system that does exactly what this project proposes.** But the project is not in greenfield territory either. The space splits into four distinct research/tooling lineages, each adjacent to this project but with different goals or constraints:

1. **AutoML / hardware-aware NAS** — automated architecture and quantization search for edge devices. Mature field, going back to ~2018. Closest to Mode A of this project, but lacks the agent and research-ingestion components.
2. **LLM-driven AutoML / NAS** — recent (2024–2026) work where LLMs guide the search rather than handcrafted optimizers. Most relevant single line of prior work; directly informs this project's Search Strategist Agent design.
3. **AI-Scientist-style fully-autonomous research systems** — closest in *agent topology* to what this project proposes, but aimed at producing papers rather than deployable models, and operating without the safety/sandbox/human-gate discipline this project requires.
4. **Edge inference frameworks with hand-tuned models** — the production systems we're effectively competing against. Liquid AI's LFM2-VL, Apple's FastVLM, MLX, llama.cpp. These are the *outputs* of human-driven optimization; this project tries to automate the process that produces them.

The novelty of this project is in the **intersection** of these four — specifically, applying LLM-driven AutoML *to VLMs*, *on real edge hardware*, with a *Mode A / Mode B escalation pattern* that includes literature ingestion as a designed-in capability rather than an autonomous-research moonshot. None of the existing systems combine all four.

This document walks through each lineage, names the most relevant projects, and ends with what this project should learn from each.

---

## 2. AutoML and hardware-aware NAS — the mature substrate

This is the longest-running and most-mature lineage. Representative works:

| Project | What it does | Relevance to this project |
|---|---|---|
| **AMC** (He et al., 2018) — *AutoML for Model Compression* | Reinforcement learning agent that prunes channels in CNNs for mobile deployment. One of the first "AutoML for edge" papers. | Conceptually similar to Mode A's compression-sweep capability, but for CNNs, not VLMs. Predates LLM-driven methods. |
| **EfficientNet-EdgeTPU / MobilenetEdgeTPU** (Gupta & Akin, 2020) | Hardware-aware NAS targeting Google Edge TPU. Codesigns model architecture with the accelerator. | Direct precedent for the "DeviceDescriptor + hardware-aware search" idea in this project. Different accelerator (Edge TPU vs. ANE/MLX), different model family (CNN vs. VLM). |
| **JASQ** (Joint Architecture Search + Quantization, 2018) | Multi-objective evolutionary search over (architecture, quantization) jointly. | Same problem shape as Mode A — but evolutionary rather than LLM-driven, and CNN-focused. |
| **BatchQuant / QFA** (Apple, 2021) | Quantized-for-all NAS — single supernet trained once, quantization policy selected per-deployment. | Apple's own approach to the problem. Inspired some design in MLX. Notably, Apple shipped FastVLM later as a *hand-engineered* model, suggesting their automated approaches weren't sufficient for VLMs of this complexity. |
| **FOX-NAS** (2021) | Quantization-friendly, on-device, explainable NAS for edge CPUs. | Methodologically closest to Mode A in this project. Uses simulated annealing rather than LLM-driven exploration. |

**What this lineage establishes:** The basic shape of hardware-aware AutoML — multi-objective Pareto search over (architecture × quantization × runtime), measured on real devices, with cheap proxies before expensive on-device measurement — is well-established and not novel. Mode A in this project is essentially a re-implementation of this lineage adapted to VLMs.

**What this lineage *doesn't* do:**

- Treat the search algorithm as an LLM-driven agent.
- Ingest research literature to extend the search space.
- Address the cross-LM-and-vision-encoder joint search space VLMs introduce.
- Target *generative* VLM tasks (most prior work is CNN classification).

**Implication for this project:** Borrow the methodology aggressively in Phase 1. Specifically, the multi-objective Pareto framing, on-device measurement, cheap-proxy filtering, and supernet-style weight sharing are all proven techniques worth adopting where applicable. Don't invent new methods where existing ones work; the project's contribution is at a higher level than the search algorithm itself.

---

## 3. LLM-driven AutoML and NAS — the directly relevant lineage

This is the lineage most directly relevant to the **Search Strategist Agent** design. It's recent (mostly 2024–2026) and still rapidly evolving.

| Project | What it does | Relevance to this project |
|---|---|---|
| **MONAQ** (Multi-Objective NAS via LLM Querying, May 2025) | Uses LLM agents to design search spaces and *evaluate candidates without runtime execution*, leveraging the LLM's pretrained knowledge. Targets time-series on resource-constrained devices. | **Very close architectural pattern to this project's Search Strategist.** Key difference: MONAQ skips real on-device measurement and trusts the LLM's predictions; this project insists on real measurement. The MONAQ approach is faster but less reliable; this project trades speed for ground truth. |
| **AutoMaAS** (Self-Evolving Multi-Agent Architecture Search, Oct 2025) | NAS for *multi-agent system design itself* — uses differentiable architecture search to optimize agent topology and tool selection. | Tangentially relevant — meta-level: optimizing the agent system, not VLMs. Worth knowing exists because the techniques could be applied to our own Search Strategist's policy. |
| **Trirat et al.** (LLM-guided AutoML) | Agent-based LLM-guided AutoML with specialized agents collaborating from task descriptions to deployment-ready models. | Closest "shape" to this project's overall topology, but general-purpose AutoML, not VLM/edge specific, and lacks the Mode A / Mode B distinction. |
| **Data-Local Autonomous LLM-Guided NAS** (March 2026) | LLM-guided NAS for multimodal time-series, with data-locality / privacy as a constraint. | Same era and approach family as this project. Useful as a template for how recent papers frame this kind of work. |

**What this lineage establishes:**

- LLMs can productively guide NAS, especially via natural-language search space specifications and structured edits to configurations.
- A multi-agent topology with specialized roles (search, evaluation, ranking) is a working pattern.
- On-device or hardware-aware constraints can be incorporated, but most existing work uses LLM-predicted performance rather than real measurement.

**What this lineage *doesn't* do:**

- None target VLMs specifically. The unique challenges of VLM optimization (vision-token budget, encoder-LM joint search, multimodal evaluation noise) are not addressed.
- None use a Mode A / Mode B escalation pattern with explicit human-gated mode transitions.
- None treat literature ingestion as a designed-in capability that *extends* the search space at runtime.
- Most rely on LLM-predicted performance rather than real device measurement, which sacrifices ground truth.

**Implication for this project:** This is the most directly informative prior art. The Search Strategist Agent design should borrow LLM-as-search-policy patterns from MONAQ and the Trirat work, but commit harder to real device measurement than they do. The Mode A / Mode B distinction is, as far as I can tell, novel — and worth defending as a contribution.

---

## 4. AI-Scientist and the autonomous-research lineage

This is the most architecturally similar to our **Research Analyst Agent**, but with very different goals and a much more aggressive autonomy claim that has not aged well.

| Project | What it does | Relevance to this project |
|---|---|---|
| **AI-Scientist v1** (Sakana AI, Aug 2024) | LLM agent system that generates ML research ideas, runs experiments, writes papers. Produces a paper for ~$15. | **Closest agent topology to what this project proposes**, but aimed at producing papers, not deployable models. |
| **AI-Scientist v2** (Sakana AI, 2025) | Generalizes v1 with agentic tree search, removes reliance on human-authored templates. First workshop paper written entirely by AI accepted via peer review. Published in *Nature* in March 2026. | More mature version. Tree-search-based experiment management is relevant to our Search Strategist design. |
| **Beel, Kan & Baumgart evaluation** (Feb 2025, peer review of AI-Scientist) | Critical evaluation showing AI-Scientist produces papers with median 5 citations, mostly outdated, frequent structural errors, hallucinated numerical results. Compared its outputs to "an unmotivated undergraduate student rushing." | **Critical reading for this project.** Documents exactly the failure modes the HLD's §6.4 anticipates: citation hallucination, misreading, statistical-noise-as-signal. Confirms our LLM-as-analyst design is the right response. |
| **Bayes-Entropy Collaborative Agents** (Aug 2025) | Probabilistic framework for research hypothesis generation, addressing the quality/reliability problems documented in the Beel et al. critique. | Useful methodology for the hypothesis-quality side of our Research Analyst. |
| **AutoGen / CrewAI multi-agent research systems** (2023–2025) | General-purpose multi-agent frameworks that coordinate retriever, summarizer, synthesizer agents for literature review and research synthesis. | Tooling we could potentially adopt for the Research Analyst rather than building from scratch. Worth evaluating in the Detailed Plan. |
| **PaperQA2, STORM, LitLLM, OpenScholar** (2024) | Specialized agentic literature-review systems. | Direct precedent for the literature-ingestion portion of our Research Analyst. Worth evaluating as components rather than reinventing. |

**What this lineage establishes:**

- A multi-agent LLM system *can* produce research-like outputs end-to-end. AI-Scientist v2 has a peer-reviewed paper as evidence.
- The failure modes are well-documented: hallucination, structural errors, citation problems, confused experimental claims. These are not solved problems.
- The "template" requirement in AI-Scientist v1 (which the evaluation paper criticized as limiting autonomy) is essentially the same trade-off as our Tier 1 / Tier 2 split — fully open-ended autonomy doesn't work yet, so bounded autonomy via templates does.

**Critical difference from this project:**

AI-Scientist tries to **produce papers**. This project tries to **produce deployable models**. The validation criteria differ in a way that matters: a paper's success is judged by reviewers (subjective, slow, expensive); a deployable model's success is judged by Pareto frontier measurement on real hardware (objective, fast, cheap). This makes our project's validation loop much tighter than AI-Scientist's, which should make the LLM-driven components fail more visibly and recoverably.

**Implication for this project:**

- The Beel et al. evaluation paper is required reading. The failure modes it documents are exactly what our §6.4 must mitigate. We should explicitly cite it when writing about the LLM-as-analyst design choice — it's our best evidence that fully-autonomous research generation doesn't work yet and human-in-the-loop verification is the responsible design.
- Evaluate whether to *use* existing tools (PaperQA2, OpenScholar, AutoGen) as components of our Research Analyst rather than building it from scratch. This could save weeks of implementation work in Phase 3.
- The contrast with AI-Scientist is itself a positioning advantage for this project: "AI-Scientist tries to be a scientist; we try to be a deployment engineer. The latter is more bounded, more verifiable, and more useful."

---

## 5. Production edge inference frameworks — the competitors

This is what we're effectively competing *against* in the success criteria. These are the hand-engineered systems that produce edge VLMs without any of the agent-system infrastructure this project proposes.

| Project / model | What it is | Why it's the baseline to beat |
|---|---|---|
| **Liquid AI LFM2-VL family** | Hand-engineered edge VLMs (450M, 1.6B, 3B). SigLIP2-NaFlex vision encoder + LFM2 backbone. Sub-250ms on Jetson Orin. | The primary bar. If our system can autonomously reach Liquid's quality at comparable size, we've validated the system framing. |
| **Apple FastVLM** | Hand-engineered edge VLM (0.5B, 1.5B, 7B). FastViTHD vision encoder, Apple Silicon-tuned via MLX. Sub-120ms TTFT on iPhone 16 Pro. | Reference for what's possible on iPhone with maximum hand-engineering. Their innovation (FastViTHD's high-resolution efficient encoding) is a candidate technique for our Mode B Research Analyst to surface and propose. |
| **MLX, llama.cpp, ONNX Runtime, CoreML** | Runtime frameworks. Not models but the *deployment substrate* our system uses. | We are not building runtimes; we are choosing between them via DeviceDescriptor. These should be treated as fixed environment, not as targets of optimization. |
| **Hugging Face Optimum, torch.ao.quantization** | Compression toolkits we depend on. | Tooling, not competition. |

**What these establish:** A skilled human team can produce competitive edge VLMs in the 6–18 month timeframe. Liquid's LFM2-VL-450M was released April 2026 after the team had been iterating on the family for over a year. Apple's FastVLM was a CVPR 2025 release after years of FastViT/FastViTHD encoder development.

**Implication for this project:** The "system reaches Liquid's bar autonomously" success criterion is ambitious because Liquid's bar is the result of focused human work. The right framing is: *Liquid is the bar for a system that started with zero knowledge of edge VLMs and reached the bar through automated optimization plus literature ingestion.* That's a strictly stronger claim than "we produced a model as good as Liquid" — and a stronger argument that the system is the real product.

---

## 6. What is novel about this project

After surveying the four lineages, here is an honest accounting of what this project would actually contribute:

**Genuinely novel (no direct prior art I found):**

- **The Mode A / Mode B escalation pattern with Decision-Dossier-gated human approval.** None of the LLM-driven AutoML systems separate "exploit known techniques" from "explore new techniques from literature" with an explicit, human-gated transition. They either stay in one mode or auto-escalate.
- **Hypothesis records as implementation kits.** AI-Scientist-style systems try to have the LLM implement hypotheses directly, which fails because of code-generation unreliability. This project's design (LLM extracts structured records, humans implement Tier 2, system auto-runs Tier 1) is a different point in the design space that — to my reading of the literature — hasn't been explicitly tried for ML optimization.
- **VLM-specific search space with vision-token-budget as a first-class knob.** Most LLM-driven NAS targets single-modality models. The cross-modality search (vision encoder × language model × quantization × tokenization × runtime) is not addressed in current systems.
- **Two-Mac asymmetric topology with research-vs-execution physical separation.** This is a specific deployment architecture, not a research contribution, but it's not been documented in this form anywhere.

**Synthesizes existing ideas in a new combination:**

- Hardware-aware NAS (from the AutoML lineage) + LLM-driven search (from §3) + literature ingestion (from §4) + real on-device measurement (largely from §2, less from §3) = the overall system.
- DeviceDescriptor as a parameter making hardware/runtime first-class is a synthesis of accelerator-aware NAS practices, not a new idea, but the implementation as YAML configs read by both agents and services is cleaner than what I've seen in academic prior art.

**Not novel but worth doing well:**

- Sandboxing, schema validation, canary runs, golden-set regression gates — all standard ML-ops practices.
- VLMEvalKit-based evaluation — using existing tooling.
- Pareto frontier maintenance — textbook multi-objective optimization.

### 6.1 Phase-2 construction & distillation — the methods are well-known and general *(added 2026-06-16)*

This document was first written for the Phase-1 *search/NAS* framing. The Phase-2 pivot (ADR-0011/0012) moved the work to **constructing and distilling a right-sized student** (HLD §6.5). For the avoidance of doubt: **every training-time method the construction loop uses is a standard, published technique applied in its conventional form.** None are a contribution of this project — the contribution is the *system that proposes, builds, evaluates, and re-routes over them autonomously*. The methods, with references:

| Method (as used here) | What it is | Reference |
|---|---|---|
| **Knowledge distillation** | Train a small *student* to reproduce a larger *teacher*'s behavior. | Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*, arXiv:1503.02531 |
| **Sequence-level KD** (teacher generates answer targets; student imitates — exactly our cached `{image, prompt, target}` recipe) | Distill on the teacher's *generated sequences* rather than its soft logits. | Kim & Rush (2016), *Sequence-Level Knowledge Distillation*, EMNLP; arXiv:1606.07947 |
| **LoRA** (the distill-stage adapter) | Parameter-efficient fine-tuning via low-rank weight updates; we use the stock `peft` implementation. | Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685 (ICLR 2022) |
| **LLaVA-style assembly + two-stage training** (vision encoder + MLP projector + LM; *align projector* then *distill*) | The standard open-VLM architecture and its projector-align-then-instruction-tune recipe. | Liu et al. (2023), *Visual Instruction Tuning*, arXiv:2304.08485 (NeurIPS 2023) |
| **SigLIP vision encoder** | The pretrained image encoder the student is built on. | Zhai et al. (2023), *Sigmoid Loss for Language Image Pre-Training*, arXiv:2303.15343 (ICCV 2023) |
| **Catastrophic forgetting** (the failure mode that broke P2-D1/D2 and the MCQ pilot) | A network loses a prior skill when trained on a new one. | McCloskey & Cohen (1989), *Catastrophic Interference in Connectionist Networks* |
| **Rehearsal / experience replay** (our fix: replay prior-skill data while learning a new one) | Mix old-task examples into new-task training to mitigate forgetting. | Robins (1995), *Catastrophic Forgetting, Rehearsal and Pseudorehearsal*, Connection Science 7(2):123–146 |

The *teacher* (Qwen2.5-VL-3B), the *student* backbones (Qwen2.5-0.5B, SigLIP), the reference/yardstick models (LFM2-VL, SmolVLM, MiniCPM-V, FastVLM), and the benchmarks (POPE, MMBench, RealWorldQA) are all third-party and used, not redistributed — their sources and licenses are in [`THIRD_PARTY.md`](THIRD_PARTY.md). Method provenance for the construction loop is summarized here; the HLD (§6.5) describes only the *mechanism*, not the method lineage.

---

## 7. What this project should learn from prior art

Concrete recommendations based on the survey:

**R1. Adopt MONAQ's LLM-search-space-design pattern in the Search Strategist Agent.** Their approach of having the LLM define and refine the search space, rather than just sampling from a fixed space, is the most relevant single design pattern to borrow.

**R2. Evaluate existing literature-ingestion tools before building the Research Analyst Agent from scratch.** PaperQA2, OpenScholar, AutoGen-based research systems exist and may be wrappable. The hypothesis-record format from §6.2 of the HLD is novel; the literature-fetching-and-summarization plumbing under it is not.

**R3. Cite Beel et al. (Feb 2025) as the evidence base for our LLM-as-analyst design choice.** That paper's critical evaluation of AI-Scientist is the cleanest published evidence that fully-autonomous research generation has documented failure modes our design specifically addresses. It strengthens the project's external positioning.

**R4. Position this project explicitly against AI-Scientist as a contrast.** "AI-Scientist tries to be a research scientist generating papers; we try to be a deployment engineer producing models. The latter is more bounded, more verifiable, and more useful as a system." This is a clean positioning statement for external audiences (advisors, reviewers, potential collaborators).

**R5. Treat the AutoML/NAS lineage's methodological contributions (supernet-based weight sharing, hardware-aware accuracy predictors, evolutionary search) as candidate techniques for the Mode B Research Analyst to surface — not as things we need to reinvent.** Many of these would be Tier 1 (config-change-only) hypotheses if the search space is set up correctly.

**R6. Acknowledge that hardware-aware NAS (in some form) is the baseline being matched, not the contribution.** The project's contribution is the *system that produces hardware-aware optimizations across tasks and devices*, not the hardware-aware optimization itself. Mode A is intentionally a re-implementation of a known technique; Mode B is where the novelty lives.

---

## 8. Open questions surfaced by this analysis

**Q1: Should we evaluate building on AI-Scientist v2's codebase rather than from scratch?** It's open-source (SakanaAI/AI-Scientist-v2). The agentic tree-search pattern they use is sophisticated. The downside is that AI-Scientist's failure modes are extensively documented; adopting their codebase means adopting their bugs. Worth a Phase 0 spike to evaluate.

**Q2: Should we evaluate MONAQ's code if available?** Their pattern of LLM-as-search-space-designer is the closest single technique we'd want to borrow.

**Q3: Should the project's external framing emphasize the contrast with AI-Scientist?** If the external positioning is going to advisors, reviewers, or potential collaborators, "we're the AI-Scientist for deployment, with safety rails AI-Scientist didn't have" is a compelling pitch. If the external framing is going to industry product reviewers, AI-Scientist is unknown to them and the framing should center on Liquid AI / Apple FastVLM comparison instead. The right framing depends on audience.

**Q4: Is the "DeviceDescriptor + cross-runtime + multi-device" framing worth highlighting as a research contribution?** Most LLM-driven AutoML targets a single hardware platform. The fact that this project optimizes for iPhone *and* Pi 5 in a single search loop is methodologically distinctive. Worth surfacing in any external write-up.

---

## 9. Summary

The project is not in territory anyone has fully claimed, but every individual capability it builds has prior art. The novelty is in the **combination and operational discipline**: LLM-driven AutoML, applied to VLMs, on real edge hardware, with literature ingestion as an explicit Mode B capability, with hypothesis records as implementation kits rather than autonomous implementations, with human-gated mode escalation via Decision Dossiers, and with safety/sandbox/canary discipline that the closest prior art (AI-Scientist) lacks.

The most useful single piece of prior reading is the Beel et al. (Feb 2025) evaluation of AI-Scientist — it documents the failure modes our HLD specifically addresses, and citing it strengthens the case for our design choices.

The most useful single piece of prior code to evaluate borrowing from is AI-Scientist v2 (for the agentic tree search) and PaperQA2 / OpenScholar (for literature ingestion plumbing). Building these from scratch in Phase 3 is possible but probably not the best use of time.

---

*This analysis informs the Goals document (in particular, the project's external positioning) and the Detailed Plan (specifically, which existing tools to evaluate adopting versus building). Both documents should be updated to reflect the findings once they are reviewed.*
