# ADR-0001: Mac Measurement Methodology

**Date:** 2026-05-20
**Status:** Accepted
**Context:** Task 2.1 — Qwen2.5-VL-3B baseline on Mac mini (M4, 16 GB)

---

## Context

We need a repeatable methodology for measuring VLM inference on the Mac mini
(and later the M5 Pro 32 GB). The methodology must produce numbers that are:

- Comparable across runs (same warmup policy, same memory metric, same TTFT definition)
- Honest about reliability (swap contamination, thermal state)
- A template for the iPhone (ADR-0002) and Pi 5 (ADR-0003) methodologies

---

## Decisions

### 1. Memory metric: `torch.mps.driver_allocated_memory()`, not `psutil` RSS

**Rejected:** `psutil.Process().memory_info().rss`

On Apple Silicon, model weights are loaded into the MPS (Metal Performance Shaders)
memory pool, which is in unified physical memory shared between CPU and GPU. The
process RSS reported by `psutil` reflects only the CPU-side allocations and
dramatically undercounts model memory (442 MB RSS vs 7957 MB actual for
Qwen2.5-VL-3B FP16).

**Accepted:** `torch.mps.driver_allocated_memory()`

This returns the total bytes allocated by the MPS driver, including all model
weights and activation buffers. For Qwen2.5-VL-3B FP16 it reports ~7957 MB,
consistent with the theoretical weight size (~6 GB) plus KV cache and activations.

### 2. TTFT measurement: `LogitsProcessor` hook, not streamer

**Rejected:** `TextIteratorStreamer` in a background thread

Streamer-based TTFT measures when Python receives the first decoded *text chunk*,
which lags the actual first token generation by tokenizer decode overhead and
thread scheduling jitter.

**Accepted:** `TTFTTimer(LogitsProcessor)`

A custom `LogitsProcessor` is called by the model's generation loop at the moment
each token's logits are computed. Recording `time.perf_counter()` on the first
call gives true TTFT: wall time from `model.generate()` call to first token
produced (includes image encoding and full prefill).

### 3. Warmup policy: one pass discarded, then measure

The first inference after model load incurs MPS kernel JIT compilation. In the
Task 2.1 run the warmup TTFT was 1757 ms vs a steady-state median of 1400 ms —
a 25% inflation. All reported numbers exclude the warmup pass.

The warmup uses `max_new_tokens=50` (vs 150 for measurement) to save time.

### 4. Decode throughput: tokens after TTFT divided by decode wall time

```
decode_tps = (n_generated - 1) / (t_end - first_token_time)
```

`n_generated - 1` excludes the first token (counted in TTFT). `t_end` is the
wall time when `model.generate()` returns. This measures steady-state decode
throughput only.

### 5. Aggregation: median of N images

Mean is sensitive to outliers (e.g. the 495 ms TTFT for img3, a simple kitchen
scene with fewer vision tokens than complex street scenes). Median over 5 images
is reported. All per-image results are logged; the MetricsReport stores the median.

### 6. Swap contamination threshold: 3000 MB

macOS always maintains a pool of compressed memory that registers as swap (~1–2 GB
at idle on a 16 GB machine). This does not affect inference because the model is
entirely in the MPS memory pool.

Threshold rule (encoded in `scripts/retest_task_2_1.sh`):
- `swap_used_mb ≤ 3000 MB`: acceptable; note in output, continue automatically
- `swap_used_mb > 3000 MB`: warn; require explicit `y` to continue

The `HardwareFingerprint.swap_contaminated` property in the schema still returns
`True` for any non-zero swap (correct for Pi 5, where any swap is fatal). For Mac,
callers should use `swap_used_mb > 3000` as the reliability filter.

### 7. Offline model loading for reruns

After the initial download, `--offline` (`local_files_only=True` in
`from_pretrained`) skips the HuggingFace Hub etag check. This avoids a misleading
"Fetching files" progress bar and removes a network dependency from reruns.

---

## Observed baseline (Task 2.1)

Device: Mac mini M4, 16 GB unified memory
Model: `Qwen/Qwen2.5-VL-3B-Instruct`, FP16, MPS backend, greedy decode
Images: 5 COCO val2017 photos, 150 max new tokens, median of 5 runs

| Metric | Value |
|---|---|
| TTFT (median) | 1400 ms |
| Decode throughput (median) | 11.1 tok/s |
| Peak memory (MPS driver) | 7957 MB |
| On-disk size | 7172 MB |
| swap_used_mb at measurement | ~1954 MB (macOS baseline) |
| Model load time (cached) | ~21 s |

TTFT range across 5 images: 495 ms (simple scene) – 1449 ms (complex scene).
Variation reflects prefill cost scaling with vision token count.

Pi 5 non-fit: 3B FP16 ≈ 7.2 GB weights alone > Pi 5 4 GB total RAM.
No attempt made; confirmed by arithmetic.

---

## Applicability to future devices

| Decision | iPhone (ADR-0002) | Pi 5 (ADR-0003) |
|---|---|---|
| Memory metric | `os_proc_available_memory()` or `mach_task_basic_info` | `psutil` RSS (no GPU pool) |
| TTFT | Same `LogitsProcessor` approach if using Python runtime | llama.cpp timing API |
| Warmup | Same policy (1 pass discarded) | Same policy |
| Swap threshold | N/A (iOS has no swap) | 0 MB — any swap is disqualifying |
