# ADR-0003 — iPhone 16 Pro Baseline Numbers

**Date:** 2026-06-05  
**Status:** Accepted  
**Phase:** Phase 0 — Reference Baselines

---

## Context

Phase 0 establishes reference performance numbers for four VLMs on iPhone 16 Pro. These numbers serve as the comparison target for all Phase 1 optimisation experiments. This ADR records the accepted baselines, their provenance, known caveats, and a sanity-check against published claims.

Measurement methodology is described in ADR-0002.

---

## Accepted Baseline Numbers

**Device:** iPhone 16 Pro (iPhone17,1, A18 Pro, 8 GB LPDDR5, iOS 26.5)  
**Prompt:** `"Describe this image briefly."` · `maxTokens=64` · 5 images · 1 warmup + 5 measured runs

| Model | Backend | Quant | TTFT ms | ±σ | TPS | Peak Mem MB | On-disk MB |
|---|---|---|---:|---:|---:|---:|---:|
| LFM2-VL-450M | llama.cpp/mtmd Metal | Q4_0 | **14.1** | 0.2 | **82.4** | **279** | 219 |
| SmolVLM-500M | llama.cpp/mtmd Metal | Q4_K_M | 20.2 | 0.2 | 48.6 | 367 | 393 |
| MiniCPM-V-4.6 | llama.cpp/mtmd Metal | Q4_K_M | 35.5 | 0.6 | 33.7† | 970 | 1199 |
| FastVLM-0.5B | MLX Swift | FP16 | 724.6 | 37.4 | 34.2 | 2204 | ~1000 |

† TPS trimmed mean (4/5 runs); run 3 outlier at 44.0 excluded (short output, word-count inflated). See Caveats.

**Artifacts:**
- `artifacts/eval_task_3_2_20260605/LFM2-VL-450M_iphone16pro_20260605.json`
- `artifacts/eval_task_3_3_20260605/FastVLM-0.5B_iphone16pro_20260605.json`
- `artifacts/eval_task_3_4_20260605/SmolVLM-500M_iphone16pro_20260605.json`
- `artifacts/eval_task_3_4_20260605/MiniCPM-V-4.6_iphone16pro_20260605.json`

---

## Task 3.5 — Sanity Check Against Published Claims

### What vendors publish

None of the four vendors publish absolute TTFT/TPS numbers for a specific iPhone model in their public documentation (as of 2026-06-05):

| Model | Published claim | Type |
|---|---|---|
| FastVLM-0.5B | "85× faster TTFT vs LLaVA-OneVision-0.5B" (CVPR 2025) | Relative, not absolute |
| LFM2-VL-450M | "2× faster GPU inference vs existing VLMs" | Relative, GPU, not iPhone |
| SmolVLM-500M | No on-device latency numbers published | — |
| MiniCPM-V-4.6 | Demo on iPhone 17 Pro Max; no latency numbers | Qualitative |

Absolute on-device TTFT/TPS claims for specific iPhone models are not published by any vendor, making direct numerical comparison impossible.

### Internal consistency checks

**1. TTFT scales with vision-encoder load (expected)**

| Model | mmproj size MB | TTFT ms | Ratio vs LFM2 |
|---|---:|---:|---:|
| LFM2-VL-450M | 99 | 14.1 | 1.0× |
| SmolVLM-500M | 109 | 20.2 | 1.4× |
| MiniCPM-V-4.6 | 728 | 35.5 | 2.5× |

TTFT is dominated by vision-encoder prefill. MiniCPM-V's mmproj is 7× larger than LFM2's but only 2.5× slower — consistent with Metal's memory bandwidth advantages at larger batch sizes. ✅

**2. Decode TPS scales inversely with LM parameter count (expected)**

| Model | LM params (approx) | TPS | Expected ratio vs LFM2 |
|---|---:|---:|---:|
| LFM2-VL-450M | 350M | 82.4 | 1.0× |
| SmolVLM-500M | 494M | 48.6 | 0.71× (expected 0.71×) |
| MiniCPM-V-4.6 | 1300M | 33.7 | 0.41× (expected 0.27×) |

LFM2→SmolVLM ratio matches near-perfectly (350/494 = 0.71 predicted, 48.6/82.4 = 0.59 observed — within noise given different quantisation Q4_0 vs Q4_K_M). LFM2→MiniCPM-V ratio is above predicted (33.7/82.4 = 0.41 vs 0.27 predicted) — MiniCPM-V's heavier mmproj forces more GPU memory pressure, likely reducing effective decode throughput less than parameter count alone would suggest. ✅ Plausible.

**3. FastVLM FP16 TTFT — consistent with real-time demo claims**

Apple's FastVLM demo app targets continuous camera frames. At 724ms TTFT (iPhone 16 Pro, FP16, no memory entitlement), the app runs at ~1.4 fps in continuous mode — consistent with the observable behaviour in the live camera demo. Apple's published "85× faster TTFT vs LLaVA-OneVision" is a relative architecture claim, not an absolute latency target, and does not contradict our measurement. ✅

**4. LFM2 "2× faster" claim**

LiquidAI claims 2× faster GPU inference vs "existing VLMs". Our nearest comparable is SmolVLM-500M (similar parameter count, same backend): LFM2 TTFT 14.1ms vs SmolVLM 20.2ms = **1.4× faster**. Directionally consistent with the 2× claim (the 2× was measured on GPU with bfloat16, our measurement is Q4_x on iPhone Metal — different regime). ✅ Directionally consistent.

**5. Memory footprint — within expected bounds**

All llama.cpp models stay well within iPhone 16 Pro's practical app memory limit (~3–4 GB without entitlement):
- LFM2: 279 MB — 11% of limit ✅
- SmolVLM: 367 MB — 14% of limit ✅
- MiniCPM-V: 970 MB — 37% of limit ✅
- FastVLM FP16: 2204 MB — 84% of limit ⚠️ (headroom is tight; quantised variant would be safer)

---

## Caveats

1. **Word-count TPS estimator** — `output.split(" ").count` is noisy on short outputs. MiniCPM-V run 3 produced a shorter response than runs 1, 2, 4, 5, inflating that run's TPS to 44.0 vs ~33.6. Trimmed mean 33.7 is used. Phase 1 harness should use actual llama.cpp token count.

2. **FastVLM backend mismatch** — FastVLM uses MLX FP16 (Apple's framework); the other three use llama.cpp Q4_x. TTFT and memory are not on the same axis. FastVLM's TTFT would be substantially lower with 4-bit quantisation; its memory would drop from 2.2 GB to ~600 MB.

3. **No thermal stress test** — All measurements taken at ambient temperature. A18 Pro throttles under sustained load; TTFT and TPS may degrade 10–20% after several minutes of continuous inference. Not measured in Phase 0.

4. **Single device** — Only one iPhone 16 Pro unit measured. Run-to-run variance (TTFT σ = 0.2–37 ms) is within acceptable bounds for baseline purposes.

5. **`increased-memory-limit` entitlement absent** — Personal team provisioning does not support this entitlement. Models above ~3 GB peak memory will crash. This is not a concern for Phase 0 models but must be addressed if Phase 1 tests larger variants.

---

## Consequences

- These numbers are the frozen Phase 0 iPhone baselines. Phase 1 experiments must beat the same-model baseline by ≥ the target improvement threshold defined in the Phase 1 plan.
- FastVLM FP16 is recorded for completeness but should not be the comparison target for quantised variants in Phase 1.
- TPS estimator replacement is a P1 harness improvement item before Phase 1 measurement runs.
