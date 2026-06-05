# Building a Multi-Agent VLM Optimization System — Phase 0 Complete

*Draft — to be published with Phase 1 reveal when the repository goes public.*

---

I've spent the past five weeks building the foundation for a system that automatically optimizes vision-language models for iPhone deployment. This is the story of Phase 0: what I measured, what surprised me, and what comes next.

---

## The problem

Small vision-language models (VLMs) — models that can look at an image and answer questions about it — are getting good enough to run on phones. Models like LFM2-VL-450M, SmolVLM-500M, MiniCPM-V-4.6, and FastVLM-0.5B all run on an iPhone 16 Pro today. But "runs" and "runs well" are different things.

The gap between a model that technically produces output and one that's fast enough to feel real-time is enormous. I wanted to understand that gap precisely, and then close it — systematically, with experiments tracked, decisions documented, and results reproducible.

Phase 0 is the measurement phase. Before optimizing anything, you need to know exactly where the bar is.

---

## What I measured

### Four models on iPhone 16 Pro

I built a native iOS harness in Swift + ObjC++ wrapping `llama.cpp`'s multimodal inference library, deployed it to an iPhone 16 Pro (A18 Pro, iOS 26.5), and ran five measured inference passes per model over five test images.

| Model | Backend | Quant | TTFT ms | TPS | Peak Mem MB |
|---|---|---|---:|---:|---:|
| LFM2-VL-450M | llama.cpp Metal | Q4_0 | **14** | **82** | **279** |
| SmolVLM-500M | llama.cpp Metal | Q4_K_M | 20 | 49 | 367 |
| MiniCPM-V-4.6 | llama.cpp Metal | Q4_K_M | 36 | 34 | 970 |
| FastVLM-0.5B | MLX Swift | FP16 | 725 | 34 | 2204 |

TTFT is time-to-first-token — the latency the user experiences before anything appears on screen. At 14ms, LFM2 feels instant. At 725ms, FastVLM has a noticeable pause before it starts generating.

**The FastVLM result was the biggest surprise.** Apple's FastVLM paper claims 85× faster TTFT than LLaVA-OneVision — and that's a real architectural achievement. But the paper measures on a GPU with optimized quantized weights. On iPhone via MLX with FP16 weights (no 4-bit quantized MLX model exists yet for the 0.5B variant), the model has to load ~1GB of FP16 weights before the first token, costing 700ms. The architecture is fast; the runtime configuration isn't.

The three llama.cpp models cluster tightly at 14–36ms TTFT, all using the same Metal backend with 4-bit quantized weights. LFM2 leads because its vision encoder projector (99MB) is the smallest — TTFT is dominated by vision-encoder prefill cost, and LFM2 does less of it.

### Five models on Mac for quality benchmarks

On Mac mini M4 (16GB), I ran all five reference models — including Qwen2.5-VL-3B as the Phase 2 starting point — through three standard benchmarks at 100 samples each:

| Model | POPE % | RealWorldQA % | MMBench % |
|---|---:|---:|---:|
| Qwen2.5-VL-3B | 97 | 55 | 66 |
| MiniCPM-V-4.6 | 92 | **65** | **79** |
| LFM2-VL-450M | 92 | 42 | 74 |
| SmolVLM-500M | 90 | 42 | 66 |
| FastVLM-0.5B | 87 | 37 | 53 |

MiniCPM-V-4.6 is the quality leader among the sub-500M models despite not leading on latency. FastVLM scores lowest on every structured benchmark.

But structured benchmarks only test yes/no and multiple-choice. They don't measure whether a model writes a *good description* of a photo — which is the actual use case for a mobile VLM assistant.

### CLIP-score: description quality

I also ran all four models on an open-ended description task ("Describe what you see in this image") and scored each output using CLIP-score — a reference-free metric that measures semantic alignment between an image and its generated description.

| Model | CLIPScore | ±σ |
|---|---:|---:|
| MiniCPM-V-4.6 | **28.3** | 3.7 |
| LFM2-VL-450M | 27.6 | 3.5 |
| FastVLM-0.5B (iPhone) | 27.1 | 3.1 |
| SmolVLM-500M | 24.1 | 2.6 |

All four cluster within 4 points. FastVLM's iPhone descriptions — despite its low MCQ scores — are semantically as accurate as LFM2's Mac descriptions. Its MCQ weakness isn't a description quality problem; it's a formatting problem (the model outputs letters differently than the benchmark parser expects).

---

## The measurement infrastructure

Building this took longer than the measurements themselves. A few things I had to solve:

**iOS provisioning.** Getting a model running on a physical iPhone via a personal Apple developer account means fighting with entitlements, bundle IDs, and `ApplicationVerificationFailed` errors. FastVLM requires the `increased-memory-limit` entitlement — which isn't available on personal team accounts. I stripped it and ran with Metal's default memory limit instead, which fit for the FP16 model at 2.2GB but would fail for anything larger.

**Chat templates.** Each model has a different prompt format. LFM2 and MiniCPM-V use ChatML (`<|im_start|>user`). SmolVLM uses Idefics3 format (`User:<image>\n...<end_of_utterance>\nAssistant:`). Getting this wrong produces garbled output or silence. I parameterised the iOS harness with a `chatTemplate` field so switching models doesn't require code changes.

**Word-count TPS estimator.** The iOS harness estimated decode tokens-per-second from `output.split(" ").count`. This works fine for long outputs but breaks on short ones — one MiniCPM-V run produced a 5-word response and reported 44 t/s instead of the ~34 t/s seen in the other four runs. Phase 1 will replace this with the actual token count from llama.cpp's decode loop.

**transformers 5.x compatibility.** The Mac eval runner uses HuggingFace's `transformers` library, which released a major version between when I started and when I needed to use some models. `AutoModelForVision2Seq` was removed; SmolVLM needed `SmolVLMForConditionalGeneration`. MiniCPM-V needed three separate monkey-patches. FastVLM's `LlavaProcessor` crashed with a `//` operator failure when `patch_size=None`. I fixed all of these, but it was a two-day distraction.

---

## The eval set

Phase 1 experiments need a frozen evaluation set — a fixed collection of images, captions, and questions that every experiment is measured against. I assembled Stage A from COCO val2017:

- **100 photos** — 95 from COCO val2017 (diversity-stratified: indoor, outdoor, animals, vehicles, food, people), plus 5 existing baseline images
- **50 reference captions** — longest human-written COCO caption per image (≥40 chars)
- **45 VQA pairs** — pulled from COCO VQA v2 (214k human Q/A pairs, agreement ≥ 4/10 annotators), no manual writing required

The set is hash-pinned via SHA-256 of every file. The manifest hash goes into every `ExperimentConfig` so experiment results are always tied to the exact eval set version that produced them.

---

## What I learned

**1. LFM2-VL-450M is the most deployable model today.** It leads on every latency metric, uses the least memory, and its quality scores are competitive with larger models. For a Phase 1 optimization target, it's the strongest starting point.

**2. FastVLM's architecture advantage doesn't survive the FP16 runtime.** The paper result is real, but it requires a quantized MLX model that doesn't exist yet for the 0.5B variant. When Phase 1 produces a 4-bit MLX build, FastVLM should close dramatically on TTFT.

**3. MCQ benchmark scores and description quality are not the same thing.** FastVLM scores last on POPE/RealWorldQA/MMBench but produces CLIP-scores comparable to LFM2. Phase 1 will use CLIP-score as the quality guard, not just MCQ accuracy.

**4. iOS measurement is harder than Mac measurement by a factor of 5.** Three of the five weeks were dominated by iOS-specific problems. This is worth knowing before anyone else tries to replicate this setup.

---

## What's next

Phase 1 will build the **Mode A optimization loop**: a multi-agent system where a hypothesis-generation agent proposes optimization experiments (quantization configs, pruning schedules, distillation settings), a measurement agent runs them on-device, and a synthesis agent decides which results are worth pursuing.

The Phase 0 baselines are the target. Every Phase 1 experiment must beat the same-model baseline while keeping CLIP-score above the Phase 0 value. The eval set manifest hash is the contract.

The repository will go public at the end of Phase 1 — when there's something worth showing, not just a measurement spreadsheet.

---

*All measurements: iPhone 16 Pro (iPhone17,1, A18 Pro, 8GB, iOS 26.5). Mac mini M4 16GB. llama.cpp mtmd Metal backend. Code and full methodology in the repo.*
