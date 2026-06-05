# ADR-0002 — iOS Measurement Methodology

**Date:** 2026-06-05  
**Status:** Accepted (amended Phase 1 Task 1.1)  
**Phase:** Phase 0 — Reference Baselines

---

## Context

We need repeatable, comparable performance measurements for VLMs running on iPhone 16 Pro. The measurements must support fair cross-model comparison and be trustworthy enough to act as baseline targets for Phase 1 optimisation work.

Two harnesses were used in Phase 0:

| Harness | Models | Backend |
|---|---|---|
| `ios_harness/VLMHarness` (ObjC++ + Swift) | LFM2-VL-450M, SmolVLM-500M, MiniCPM-V-4.6 | llama.cpp / mtmd, Metal |
| `vendor/ml-fastvlm` (Swift + MLX) | FastVLM-0.5B | MLX Swift, Metal |

---

## Decisions

### 1. Metrics collected

| Metric | Definition | Unit |
|---|---|---|
| TTFT | Wall time from start of `mtmd_helper_eval_chunks` (or equivalent) to callback for the first decoded token | ms |
| Decode TPS | Total output tokens ÷ wall time from first token to last token | t/s |
| Peak memory | `task_vm_info.phys_footprint` high-water mark sampled per decode step | MB |
| On-disk size | Sum of GGUF file sizes (LM + mmproj) copied into the app bundle | MB |

### 2. Warm-up

One warm-up run (results discarded) is executed before the 5 measured runs. This amortises:
- Model weight paging from flash to DRAM
- Metal shader compilation (first run only)
- KV-cache initialisation

Without warm-up, first-run TTFT is 2–5× higher and would contaminate the mean.

### 3. Image set

All models are evaluated on the same 5 images (`sample1–5.jpg`, copied from `datasets/stage_a_proxy/photos/`). This controls for variable TTFT caused by different image content / token count.

Images are bundled inside the app binary (`.gguf` via `Copy Bundle Resources`), so there is no I/O latency between runs.

### 4. Prompt

`"Describe this image briefly."` with `maxTokens = 64`. A short, open-ended prompt minimises prompt-length variance across models. 64-token cap keeps decode time bounded and consistent.

### 5. Token counting

**Phase 0 (known limitation, now fixed):** Decode TPS was estimated using a word-count proxy, which introduced noise on short outputs. Observed outlier: MiniCPM-V-4.6 run 3 = 44.0 t/s vs ~33.6 for other runs due to shorter sample3 response.

**Phase 1 fix (Task 1.1, 2026-06-05):** `LlamaVLMRunner.mm` now uses `llama_perf_context()` for authoritative TPS:

```objc
const struct llama_perf_context_data perf = llama_perf_context(_ctx);
tps = (double)perf.n_eval / (perf.t_eval_ms / 1000.0);
```

`perf.n_eval` = tokens decoded by the Metal kernel (not word/char count).  
`perf.t_eval_ms` = cumulative Metal+CPU time for those decode steps.  
`llama_perf_context_reset(_ctx)` is called before each inference to prevent cross-run accumulation.

A fallback to the legacy wall-clock formula is retained if perf counters return zero.

**Phase 1 validation required:** re-run LFM2-VL-450M × 5 runs to confirm TPS is stable across outputs of different lengths. If corrected TPS differs materially from Phase 0 (82.4 t/s), update ADR-0003.

### 6. Memory measurement

`task_vm_info.phys_footprint` measures the process's physical memory footprint as reported by the kernel, including all mapped GGUF pages and Metal buffers. It is sampled every decode step; the maximum across all steps is reported.

The baseline (before model load) is not subtracted — the reported value is the total footprint of the running inference, which is the operationally relevant number for OOM risk assessment.

### 7. Statistics

5 measured runs per model per image set. Reports include:
- Mean TTFT across runs
- Std-dev TTFT across runs (indicator of thermal/scheduling variance)
- Mean decode TPS
- Mean peak memory

No median or percentile statistics in Phase 0; sufficient for baseline purposes.

### 8. Device state

- iPhone 16 Pro (iPhone17,1, A18 Pro, iOS 26.5)
- Screen on, app in foreground, idle timer disabled
- No active background apps beyond OS services
- No thermal stress before measurement (device at ambient temperature)
- No `com.apple.developer.kernel.increased-memory-limit` entitlement (personal team constraint)

### 9. Quantisation

llama.cpp models use Q4_0 (LFM2) or Q4_K_M (SmolVLM, MiniCPM-V) as available from community GGUF repos. mmproj files use Q8_0. FastVLM uses FP16 via MLX (no quantised version available for 0.5B at time of measurement).

Quantisation differences mean raw TPS numbers are **not directly comparable** between llama.cpp and MLX models. TTFT comparisons should account for different weight sizes.

---

## Consequences

- Measurements are reproducible: same app, same images, same prompt, same warm-up protocol
- TPS now uses `llama_perf_context()` — authoritative kernel token count, not word count (Phase 1 fix)
- FP16 FastVLM is not comparable to Q4_x llama.cpp models on TPS/memory axes; treated as a separate data point
- Without `increased-memory-limit` entitlement, iOS may kill the app above ~3–4 GB; this is not an issue for Phase 0 models but will matter for larger models in Phase 1
