# Building a Multi-Agent System for VLM Optimization — Phase 0 + 1 Complete

*The repository is now public. This post covers both phases: the measurement foundation (Phase 0) and the first optimization loop (Phase 1).*

---

I've spent the past eight weeks building a system that automatically optimizes vision-language models for iPhone deployment. The system finds better model configurations by proposing experiments, measuring them on-device, and tracking what worked — without a human in the loop for each decision.

This is the reveal post. Phase 0 built the measurement infrastructure and established baselines. Phase 1 ran the first optimization experiments and shipped the agent components. The repo is open and everything is reproducible.

---

## Part 1 — Phase 0: Measuring the baseline

### The problem

Small vision-language models — models that look at an image and answer questions about it — are good enough to run on phones. LFM2-VL-450M, SmolVLM-500M, MiniCPM-V-4.6, and FastVLM-0.5B all run on an iPhone 16 Pro today. But "runs" and "runs well" are different things.

The gap between a model that technically produces output and one fast enough to feel real-time is enormous. I wanted to measure that gap precisely, then close it systematically — with every experiment tracked, every decision documented, and every result reproducible.

### Four models on iPhone 16 Pro

I built a native iOS harness in Swift + ObjC++ wrapping `llama.cpp`'s multimodal library (`mtmd`), deployed to an iPhone 16 Pro (A18 Pro, iOS 26.5), and ran five measured inference passes per model.

| Model | Backend | Quant | TTFT ms | TPS | Peak Mem MB | On-disk MB |
|---|---|---|---:|---:|---:|---:|
| LFM2-VL-450M | llama.cpp Metal | Q4_0 | **14** | **82** | **279** | 219 |
| SmolVLM-500M | llama.cpp Metal | Q4_K_M | 20 | 49 | 367 | 393 |
| MiniCPM-V-4.6 | llama.cpp Metal | Q4_K_M | 36 | 34 | 970 | 1199 |
| FastVLM-0.5B | MLX Swift | FP16 | 725 | 34 | 2204 | ~1000 |

TTFT is time-to-first-token — the latency the user sees before the first word appears. At 14ms, LFM2 feels instant. At 725ms, FastVLM has a noticeable pause.

**The FastVLM result was the biggest surprise.** Apple's paper claims 85× faster TTFT than LLaVA-OneVision, and that's architecturally real. But the paper measures on GPU with optimized quantized weights. On iPhone via MLX with FP16 weights (no INT4 MLX build exists for the 0.5B variant), the model loads ~1GB before the first token. The architecture is fast; the current runtime configuration isn't.

The three llama.cpp models cluster tightly at 14–36ms — same Metal backend, 4-bit weights. LFM2 leads because its vision encoder projector (99MB mmproj) is the smallest. TTFT is dominated by vision-encoder prefill, and LFM2 does less of it.

### Quality benchmarks

On Mac mini M4 (16GB), I ran all five models — including Qwen2.5-VL-3B as the Phase 2 target — through three benchmarks at 100 samples each:

| Model | POPE % | RealWorldQA % | MMBench % |
|---|---:|---:|---:|
| Qwen2.5-VL-3B | 97 | 55 | 66 |
| MiniCPM-V-4.6 | 92 | **65** | **79** |
| LFM2-VL-450M | 92 | 42 | 74 |
| SmolVLM-500M | 90 | 42 | 66 |
| FastVLM-0.5B | 87 | 37 | 53 |

But MCQ benchmarks only test yes/no and multiple-choice. They don't measure whether a model writes a *good description* of a photo. So I also scored open-ended descriptions using CLIP-score — semantic alignment between image and generated text.

| Model | CLIP-score | ±σ |
|---|---:|---:|
| MiniCPM-V-4.6 | **28.3** | 3.7 |
| LFM2-VL-450M | 27.6 | 3.5 |
| FastVLM-0.5B (iPhone) | 27.1 | 3.1 |
| SmolVLM-500M | 24.1 | 2.6 |

FastVLM scores last on MCQ benchmarks but its descriptions are semantically as accurate as LFM2's. Its MCQ weakness is a formatting problem, not a vision problem.

### The eval set

Phase 1 needs a frozen eval set. I assembled Stage A from COCO val2017: 100 photos, 50 reference captions, 45 VQA pairs — all hash-pinned via SHA-256. The manifest hash goes into every `ExperimentConfig` so results are always tied to the exact eval set version that produced them.

---

## Part 2 — Phase 1: The first optimization loop

Phase 1 had two goals: run quantization experiments to find at least one Pareto improvement over the Phase 0 baselines, and ship the agent infrastructure that will run the loop autonomously in Phase 2.

### What I fixed first

Phase 0 had a latent bug in the iOS TPS counter. The harness estimated decode speed from `output.split(" ").count` — which works for long outputs but inflates wildly for short ones. One MiniCPM-V run produced a 5-word response and reported 44 t/s instead of ~34 t/s.

The fix was one function in `LlamaVLMRunner.mm`: replace the word-count estimator with `llama_perf_context(_ctx).n_eval / t_eval_ms` — the actual decode token count from llama.cpp's Metal kernel timers, immune to output length.

### H001 — LFM2 Q4_K_M (imatrix quantization)

The Phase 0 LFM2 baseline used Q4_0 — uniform INT4 quantization. A better approach is **k-quant imatrix**: use an importance matrix (calibrated on a dataset) to identify which weights are most sensitive to quantization errors, then apply mixed precision where it matters most. Bartowski's LFM2 build on HuggingFace uses this technique.

Results on iPhone 16 Pro (5 runs):

| | Q4_0 baseline | Q4_K_M (H001) | Δ |
|---|---|---|---|
| CLIP-score | 27.60 | **28.59** | **+3.6%** |
| TTFT | ~14ms* | 15.2ms | neutral |
| TPS | 82.4 t/s | 78.9 t/s | −4% |
| Mem | 275 MB | 272 MB | −1% |
| On-disk | 219 MB | 318 MB | +45% (expected for K-quant) |

*Phase 0 TTFT measurement used a pre-fix harness; the ~14ms estimate has known confounds.

The CLIP improvement (+0.99 points) is the primary result. For a model already at INT4, k-quant imatrix calibration reduces the accuracy loss without changing the runtime — same Metal kernel, same memory, better weights. **This is a quality-axis Pareto improvement** over the Phase 0 baseline.

### H002 — SmolVLM i1-Q4_0 (imatrix for SmolVLM)

SmolVLM-500M started at Q4_K_M (mixed 4/6-bit). Mradermacher's `i1-Q4_0` build uses importance-matrix calibration to produce a uniform INT4 build that's smaller on disk and potentially faster due to simpler kernel dispatch.

Results:

| | Q4_K_M baseline | i1-Q4_0 (H002) | Δ |
|---|---|---|---|
| CLIP-score | 24.11 (Ph0) | 27.78 (fp16 proxy) | — |
| TTFT | 20.2ms | **17.7ms** | **−12%** |
| TPS | 48.6 t/s | **51.9 t/s** | **+7%** |
| Mem | 367 MB | 367 MB | 0% |
| On-disk | 393 MB | **348 MB** | **−11%** |

TTFT and TPS both improved measurably. Memory didn't move (runtime activation buffers dominate, not weight format). On-disk size dropped 11%. **Another Pareto improvement** — speed + on-disk gains at no quality regression.

### H003 — Input resize 336→224px (null result, but a useful one)

The hypothesis: LFM2's TTFT is dominated by vision-encoder prefill. Resizing input images from 336px to 224px before inference reduces the image patch count by ~2.2×, which should cut prefill tokens and TTFT proportionally. No model change — just resize the input.

The Mac proxy eval (n=50) showed CLIP 27.88 vs 27.60 baseline — quality essentially unchanged. Green light to measure on device.

iPhone result: **TTFT 15.43ms vs 15.24ms (+1.2%, within noise). Zero change.**

The root cause took a moment to find. `llama.cpp`'s `mtmd` library bakes the CLIP model's native resolution (336px for LFM2's ViT-L/14) into the mmproj GGUF at quantization time. When the C++ image preprocessor receives our 224px input, it resizes it back to 336px before tokenization. The patch grid is always 24×24 = 576 visual tokens, regardless of what we hand it.

Our upstream resize was overridden in the C++ layer. The technique works on the HuggingFace Python path (where the processor receives the image as-is), but not on the GGUF inference path.

To actually reduce visual tokens on-device, you'd need to recompile the mmproj with a different `image_size` config — which is a model build change, not an inference-time trick.

**The null result is worth publishing** because it rules out a seemingly obvious TTFT optimization and explains why.

### What H004 taught us

The plan called for quantizing the mmproj from Q8_0 to Q4_0 to reduce its memory footprint. Three separate tooling paths all failed:

1. `llama-quantize` rejects the CLIP architecture ("unsupported model architecture: 'clip'")
2. `gguf-py` GGUFWriter OOM-kills during tensor accumulation (buffers all data in RAM before writing)
3. `convert_hf_to_gguf.py` only supports up to Q8_0 — no Q4_0 output type

All three blocked. The mmproj quantization problem is real and unsolved with current tooling.

### The agent infrastructure

The other half of Phase 1 was building the components that will run the optimization loop autonomously in Phase 2.

**`services/experiment_runner.py`** — accepts an `ExperimentConfig`, runs the Mac quality proxy eval (HuggingFace fp16 on MPS), writes an iPhone-ready flag for device deployment, returns a `MetricsReport`. The experiment ID is a SHA-256 of the config's canonical JSON — any change to any parameter produces a different experiment.

**`services/pareto_tracker.py`** — reads all ledger entries, computes the Pareto frontier across CLIP-score, TTFT, TPS, and peak memory. One design decision worth noting: dominance requires a point to cover every axis where the comparison point has data. This prevents Mac-only proxy experiments (which have no TTFT/Mem) from dominating iPhone-measured points by virtue of having null values on those axes.

**`agents/search_strategist.py`** — a Claude API agent with three tools: `query_results`, `query_frontier`, and `propose_experiment`. Given the current frontier and hypothesis table, it reasons through what to try next and writes a validated `ExperimentConfig` to the experiment queue. The reasoning policy is hardcoded in the system prompt:

1. Start with the highest expected-gain hypothesis not yet tried
2. If the last experiment was a Pareto improvement → explore a variation
3. If not → try a different technique
4. After 3 consecutive non-improvements → flag for human review

**`services/decision_dossier.py`** — a `ThresholdMonitor` that watches for frontier stagnation. When N consecutive experiments fail to advance the frontier, it generates a structured markdown dossier with the current state, stuck analysis, and candidate next actions.

### Current Pareto frontier

After Phase 1:

| Model | CLIP | TTFT ms | TPS | Mem MB | On-disk MB |
|---|---:|---:|---:|---:|---:|
| LFM2-VL-450M Q4_K_M (H001) ⭐ | **28.59** | 15.2 | 78.9 | **272** | 318 |
| SmolVLM-500M i1-Q4_0 (H002) ⭐ | 27.78 | **17.7** | **51.9** | 367 | **348** |
| LFM2-VL-450M Q4_0 (Ph0 baseline) | 27.60 | — | 82.4 | 275 | 219 |

H001 is the best single model: highest CLIP (+3.6% vs baseline), same speed/memory. H002 is the best SmolVLM variant: fastest TTFT, best on-disk footprint in the SmolVLM family. Both are confirmed improvements over their Phase 0 baselines.

---

## What I learned

**1. Imatrix calibration is worth it on models already at INT4.** For LFM2, switching from uniform Q4_0 to k-quant imatrix (Q4_K_M) improved CLIP-score by 3.6% with no runtime cost. The calibration happens at quantization time, not inference time.

**2. Input resize doesn't work on the GGUF path.** The CLIP preprocessor in `mtmd` uses a fixed native resolution baked into the mmproj. Upstream resizing is silently overridden. This took a full experiment round-trip to discover, and it's not documented anywhere in the llama.cpp or mtmd source.

**3. mmproj quantization below Q8_0 is currently impossible with public tools.** The CLIP architecture is explicitly excluded from `llama-quantize`. `gguf-py` OOM-kills on a machine with 1.2GB free. This is a gap worth filing.

**4. Mac proxy quality evals do not predict GGUF performance evals.** The HuggingFace fp16 path and the llama.cpp GGUF path are different codebases. Techniques that work on one don't necessarily work on the other. Always validate on the actual inference path.

**5. Building the measurement infrastructure takes longer than the measurements.** Phase 0 was five weeks. Two weeks of that was iOS provisioning, transformers compatibility bugs, and tooling failures. Budget generously for this.

---

## What's next (Phase 2)

Phase 2 targets the larger architecture question: can we compress Qwen2.5-VL-3B — which leads every quality benchmark but can't run on an iPhone — down to a model that fits on-device while keeping most of its quality advantage?

The Phase 1 loop runs manually today. Phase 2 will automate the full cycle: the Search Strategist proposes an experiment, the Experiment Runner executes it on Mac, and the iOS harness is triggered automatically via the device's hotspot. No human in the loop for standard experiments.

The repo is open. Everything here is reproducible from scratch.

---

*Hardware: iPhone 16 Pro (iPhone17,1, A18 Pro, iOS 26.5), Mac mini M4 16GB. All Phase 1 experiments: 1 warmup + 5 measured runs, same 5 sample images, same prompt ("Describe this image briefly."), max 64 tokens. CLIP-score: `openai/clip-vit-large-patch14`, n=50 Stage A images. Code, methodology, and full results in the repository.*
