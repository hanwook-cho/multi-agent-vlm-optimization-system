# Observation: Qwen2.5-VL-3B Q4_K_M is Non-Viable on iPhone (Phase 2 P2-1.4)

**Date:** 2026-06-12
**Context:** Phase 2 Task P2-1.4 — the Week-1 device-feasibility gate for the Qwen2.5-VL-3B teacher
**Decision:** Document on-device non-viability from Mac-measured evidence (no forced device deploy), proceed to Strategy B (distillation).

---

## Why no on-device measurement

P2-1.4 was a **decision gate**: does Qwen2.5-VL-3B run acceptably on iPhone? If yes → it is the Phase 2 answer (skip distillation); if no → distill. The cheaper Mac-side evidence (P2-1.2 sizes, P2-1.3 path behavior, and the Mac perf below) answers "no" with high confidence, so the gate is resolved without the device-setup cost.

This follows the project's established pattern for known-negatives: Phase 0 documented Pi 5 non-viability for FastVLM without forcing the deploy (criterion 0.2), and H003/H004 documented null/blocked results with the *why*.

---

## Mac reference measurement (llama.cpp/mtmd, M4 16GB)

`Qwen2.5-VL-3B-Q4_K_M.gguf` + `mmproj-...-f16.gguf`, `--image-min-tokens 1024`, single image + prompt:

| Metric | Value |
|---|---:|
| Prompt tokens (image + text) | **1085** (image ≈ 1024) |
| TTFT / prompt-eval (prefill) | **5,129 ms** (4.73 ms/token, 211 tok/s) |
| Decode throughput | 41.7 t/s |
| Peak memory footprint | **6.53 GB** |
| Max resident set (incl. mmap) | 8.30 GB |

For comparison, the Phase 0 edge models on iPhone: TTFT 14–36 ms, peak memory 0.28–0.97 GB.

---

## Why it cannot run on iPhone

**1. Memory exceeds the ceiling.** Peak footprint 6.53 GB. iPhone 16 Pro has 8 GB RAM; the default per-app jetsam limit is ~3–3.5 GB, and even with `com.apple.developer.kernel.increased-memory-limit` the practical ceiling is ~5–6 GB. 6.53 GB is at or above that — the app would be **OOM-killed**, most likely during the image-prefill spike. (Phase 0's FastVLM already hit the entitlement wall at 2.2 GB; this is ~3× that.)

**2. Latency is unusable even on faster hardware.** TTFT is **5.1 seconds on the Mac M4** — a more powerful GPU than the A18 Pro (more cores, higher bandwidth, no thermal throttle). On iPhone it would be slower still. The cost is structural: ~1024 image tokens prefilled through a 1.34 GB F16 vision encoder.

Both factors are inherent to Qwen2.5-VL-3B's architecture (large vision-token count + large vision encoder) and are not fixable by quantization alone — the mmproj cannot go below Q8_0 (H004), and the image-token floor is needed for grounding accuracy.

---

## Decision and implication

**Qwen2.5-VL-3B is confirmed as the Phase 2 *teacher*, not a deployable model.** The Phase 2 path is **Strategy B: distill it into a 450M-class student** that fits comfortably on-device (the edge-model envelope: <1 GB memory, <40 ms TTFT).

This quantifies the distillation target — the student must close roughly:
- **Memory:** 6.53 GB → <1 GB (~7×)
- **TTFT:** 5,100 ms → <40 ms (~130×)
while inheriting the teacher's MCQ-benchmark quality lead (P2-1.3: RealWorldQA, MMBench).

The on-device measurement remains available later if a literal iPhone number is wanted for the Phase 2 blog "before" picture, but it is not blocking.

---

## Reproduce

```bash
vendor/llama.cpp/build/bin/llama-server \
  -m models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Q4_K_M.gguf \
  --mmproj models/qwen2.5-vl-3b-gguf/mmproj-Qwen2.5-VL-3B-f16.gguf \
  --port 8077 -c 4096 -ngl 999 --image-min-tokens 1024
# send one multimodal request; read slot print_timing (prompt eval / eval) from the server log.
# peak memory via: /usr/bin/time -l vendor/llama.cpp/build/bin/llama-mtmd-cli ... (peak memory footprint).
```
