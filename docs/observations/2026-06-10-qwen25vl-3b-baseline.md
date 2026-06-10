# Observation: Qwen2.5-VL-3B CLIP-Score Baseline (Phase 2 P2-1.1)

**Date:** 2026-06-10
**Context:** Phase 2 Task P2-1.1 — measure the Qwen2.5-VL-3B teacher's open-ended caption quality before distillation
**Trigger:** Establish the Phase 2 "starting point" and validate the distillation quality signal

---

## Result

CLIP-score (`openai/clip-vit-large-patch14`), open-ended descriptions, prompt "Describe what you see in this image.", Mac MPS bf16.

**Robust paired run (n=50, same 50 COCO val2017 images from Stage A):**

| Model | CLIP | ±σ |
|---|---:|---:|
| LFM2-VL-450M (edge, Ph0 baseline) | 29.00 | 3.76 |
| **Qwen2.5-VL-3B (teacher)** | **28.56** | 3.32 |

Paired difference (Qwen − LFM2): **−0.44**, SE 0.37, paired **t = −1.19** → not significant. Qwen wins 22/50 images, LFM2 wins 28/50.

**Pilot run (n=5 proxy images), for reference:** Qwen 26.81 vs LFM2 27.60 — same direction, larger gap (small-n noise).

---

## Finding

**The Qwen2.5-VL-3B teacher has NO CLIP-score advantage over the 450M edge model — they are statistically tied.** This holds robustly at n=50 with a paired test.

This is counterintuitive for the "compress a stronger 3B down to edge" framing, but expected once you look at what CLIP-score measures:

- CLIP-score caps text at 77 tokens and rewards tight image↔caption alignment.
- Qwen2.5-VL-3B writes long, framed descriptions ("In the image, there are two cats lying on a pink couch. The first cat is positioned…"). Framing words and verbosity dilute the score.
- The edge models are trained for compact captioning — they pack more salient visual content into fewer tokens.

The teacher's genuine advantage is on **MCQ / reasoning benchmarks** (Phase 0, n=100): POPE **96.7** vs 91.7 (LFM2), RealWorldQA **55** vs 42, MMBench 66 vs 74. That is where distillation should pull the student up.

---

## Decision: distillation quality signal = MCQ benchmarks, not CLIP-score

P2-1.1 changes the Phase 2 plan's success metric:

- **Before:** distilled student should reach CLIP ≥ 29.5.
- **After:** distilled student should improve on **POPE / RealWorldQA / MMBench** toward the teacher, while not regressing CLIP-score below the Phase 0 baseline (27.6).

Rationale: you cannot distill a CLIP advantage the teacher does not have. Validating Strategy B on CLIP-score would have produced a false signal. See the revised "Success targets" section in [`VLM_Optimization_DetailedPlan_Phase2.md`](../VLM_Optimization_DetailedPlan_Phase2.md).

This is exactly what an early baseline (P2-1.1) is for — it caught a flawed success metric before a week of distillation compute was spent against it.

---

## Reproduce

```bash
# n=50 paired (Qwen + LFM2 on the same 50 COCO images)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 runners/generate_descriptions.py \
  --images datasets/stage_a/photos --out artifacts/clip_preds_n50/ \
  --models Qwen2.5-VL-3B LFM2-VL-450M --limit 50
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 runners/compute_clip_score.py \
  --images datasets/stage_a/photos --preds artifacts/clip_preds_n50/ \
  --out artifacts/clip_scores_n50/
```

Artifacts: `artifacts/clip_preds_n50/`, `artifacts/clip_scores_n50/`. Qwen2.5-VL-3B was already in the HF cache (Phase 0 Task 2.1) — no download.

## Still open

- MCQ benchmark run on Qwen2.5-VL-3B GGUF (Q4_K_M) is P2-1.3; the fp16 MCQ numbers above are from Phase 0 Task 2.2.
- iPhone feasibility of Qwen2.5-VL-3B Q4_K_M (TTFT/Mem) is the Week-1 gate (P2-1.4).
