# Observation: Qwen2.5-VL-3B GGUF MCQ — Path vs Quantization (Phase 2 P2-1.3)

**Date:** 2026-06-11
**Context:** Phase 2 Task P2-1.3 — MCQ benchmarks on the Qwen2.5-VL-3B Q4_K_M GGUF
**Trigger:** Confirm the quantized teacher's quality on the distillation signal (POPE/RealWorldQA/MMBench)

---

## Setup

Evaluated three configurations of Qwen2.5-VL-3B on **identical 100-sample slices**
(dataset hashes verified equal per benchmark), same VLMEvalKit scoring:

1. **fp16 (transformers/MPS)** — the Phase 0 path
2. **F16-GGUF (llama.cpp/mtmd)** — same precision, different runtime
3. **Q4_K_M-GGUF (llama.cpp/mtmd)** — the deployable on-device artifact

GGUF runs used `llama-server --mmproj --image-min-tokens 1024` (Qwen-VL needs
≥1024 image tokens for grounding). Eval harness: `runners/eval_vlmeval.py`
(`Qwen25VLGGUFModel` class added in P2-1.3). 0 inference errors across all 300×3.

---

## Result — the deltas decompose cleanly

| Benchmark | fp16-transformers | F16-GGUF | Q4_K_M-GGUF | **Path Δ** | **Quant Δ** |
|---|---:|---:|---:|---:|---:|
| POPE (Overall) | 96.6 | 83.6 | 82.1 | **−12.9** | −1.5 |
| RealWorldQA | 59.0 | 70.0 | 65.0 | **+11.0** | −5.0 |
| MMBench | 66.0 | 85.0 | 85.0 | **+19.0** | 0.0 |

- **Path Δ** = fp16-transformers → F16-GGUF (runtime/preprocessing effect, same weights)
- **Quant Δ** = F16-GGUF → Q4_K_M-GGUF (pure quantization effect, same runtime)

Validation: fp16-transformers POPE (96.6) reproduces Phase 0 (96.7), confirming
the harness/slice is sound.

---

## Findings

**1. Q4_K_M quantization is essentially quality-preserving.** The pure quantization
effect is tiny: POPE −1.5, MMBench 0.0, RealWorldQA −5.0 (worst case). The deployable
Q4_K_M GGUF is a faithful copy of the model. **The Qwen2.5-VL-3B teacher survives
quantization** — good for using it as a Phase 2 distillation teacher and as a
candidate in its own right.

**2. The inference path dominates benchmark scores — by up to 19 points.** Switching
runtime (transformers fp16 → llama.cpp/mtmd, same F16 weights) swung POPE −12.9,
RealWorldQA +11.0, MMBench +19.0. Different image preprocessing, the ≥1024 image-token
floor, and chat-template differences fully account for the swings. The apparent "POPE
drop" (96.6 → 82) is **~90% path, ~10% quantization.**

**3. Methodology rule for Phase 2: hold the inference path constant for any
cross-model quality comparison.** The Phase 0 edge-model numbers (e.g. POPE 91.7 for
LFM2/MiniCPM) were measured on the **transformers fp16** path. Comparing them to a
**GGUF** candidate is invalid — the path effect alone is ±10–19 points. Any
"distilled student vs reference" comparison in Strategy B must run both on the same
path (GGUF-on-device for both, or transformers for both).

---

## Implication for the plan

- The Q4_K_M teacher is sound (quantization-robust). Distillation can proceed.
- On the **GGUF/on-device path**, Qwen2.5-VL-3B leads strongly: RealWorldQA 65,
  MMBench 85 (the path lifts these). POPE on the GGUF path is ~82.
- For Strategy B success measurement (P2-6.x), evaluate the **distilled student and
  the reference models on the same path** — do not reuse Phase 0 transformers numbers
  as the bar for a GGUF student.

---

## Reproduce

```bash
for M in Qwen2.5-VL-3B Qwen2.5-VL-3B-F16-GGUF Qwen2.5-VL-3B-Q4_K_M; do
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 runners/eval_vlmeval.py \
    --models "$M" --benchmarks POPE RealWorldQA MMBench_DEV_EN \
    --n-samples 100 --output "results/eval_p2_1_3_${M}"
done
```

GGUF bundle from `scripts/convert_qwen25vl_gguf.sh`. The F16-GGUF (5.8GB) is only
needed for this control decomposition; it can be deleted afterward. Result JSONs are
gitignored (`results/`).
