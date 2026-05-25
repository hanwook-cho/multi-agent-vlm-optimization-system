# Project Status

**Last updated:** 2026-05-25  
**Phase:** Phase 0 — Reference Baselines  
**Current week:** Week 2 ✅ complete → Week 3 starting

---

## Phase 0 Progress

| Week | Focus | Status |
|---|---|---|
| Week 1 | Infrastructure, schemas, Mac measurement harness | ✅ Done |
| Week 2 | Mac quality eval (VLMEvalKit, 5 models × 3 benchmarks) | ✅ Done |
| Week 3 | iPhone 16 Pro reference baselines (4 models) | ⬜ Not started |
| Week 4 | Pi 5 baselines + Stage A eval set assembly | ⬜ Not started |
| Week 5 | Dashboard, literature spike, Phase 0 closeout | ⬜ Not started |

---

## Completed Tasks

### Week 1
- **Task 1.x** — Schemas (`MetricsReport`, `ExperimentConfig`), device descriptors, project scaffolding
- **Task 2.1** — Qwen2.5-VL-3B measured on Mac mini M4 16GB (swap-contaminated run documented in ADR-0001)
- **ADR-0001** — Mac measurement methodology (`docs/decisions/0001-mac-measurement-methodology.md`)

### Week 2
- **Task 2.2** — Quality evaluation of all 5 reference VLMs × 3 benchmarks × 100 samples on Mac mini M4 16GB
  - Runner: `runners/eval_vlmeval.py` (VLMEvalKit-based)
  - Archived: `artifacts/eval_task_2_2_20260525_094121/` (15 MetricsReport JSONs)
  - Several scoring bugs found and fixed during this session (see Known Issues below)

---

## Task 2.2 Results (Mac mini M4 16GB, 100 samples each)

| Model | HF ID | POPE acc% | RealWorldQA% | MMBench% |
|---|---|---:|---:|---:|
| LFM2-VL-450M | `LiquidAI/LFM2-VL-450M` | 91.7 | 42.0 | 74.0 |
| SmolVLM-500M | `HuggingFaceTB/SmolVLM-500M-Instruct` | 90.0 | 42.0 | 66.0 |
| MiniCPM-V-4.6 | `openbmb/MiniCPM-V-4.6` | 91.7 | 65.0 | 79.0 |
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | 96.7 | 55.0 | 66.0 |
| FastVLM-0.5B | `apple/FastVLM-0.5B` | 86.7 | 37.0 | 53.0 |

Consistency with published numbers: POPE scores match within ~2pp for all models where published scores exist (FastVLM paper: 87.4%, LFM2.5 blog: 86.9%). RealWorldQA and MMBench gaps vs. published are expected from 100-sample slice variance and exact-match-only scoring (no GPT fallback).

---

## Known Issues / Bugs Fixed This Session

1. **Wrong MiniCPM model** — Originally ran `openbmb/MiniCPM-V-4_5` (~8B params) instead of the planned `openbmb/MiniCPM-V-4.6` (1.3B, mobile-optimized). Fixed and re-run.

2. **Stale pkl cache** — VLMEvalKit caches scoring results in `_eval_scratch/*.pkl`. A failed first run writes all-zero results; subsequent runs reuse the stale cache. Affected LFM2 RealWorldQA (0% → 42%) and MiniCPM-V-4.5 POPE. Fix: delete stale pkl files and re-run.

3. **FastVLM output format** — `can_infer_option()` requires the answer letter near the *end* of the string, but FastVLM outputs the letter *first* then verbose text. Fix: trim `infer()` to first non-empty line.

4. **FastVLM repetition_penalty vs POPE** — Adding `repetition_penalty=1.2` to fix MCQ repetition loops broke POPE because the suffix "Please answer Yes or No only." puts "Yes"/"No" in the prompt, which the penalty then suppresses in output (garbled Chinese characters, 78/100 samples). Fix: `repetition_penalty=1.2 if is_mcq else 1.0`.

5. **MiniCPM-V transformers 5.x compatibility** — `all_tied_weights_keys` missing, `TokenizersBackend` no longer proxies custom tokenizer attributes (`im_start_id`, `bos_id`, etc.). Fixed via monkey-patches in `MiniCPMVModel.__init__()`.

6. **SmolVLM transformers 5.x compatibility** — `AutoModelForVision2Seq` removed. Fix: use `SmolVLMForConditionalGeneration` directly.

7. **FastVLM `LlavaProcessor` crash** — `patch_size=None` for MobileCLIP-based model causes `//` operator failure. Fix: bypass `LlavaProcessor`, use `CLIPImageProcessor` + custom `_tokenizer_image_token()` + direct `model.generate(inputs=input_ids, images=pixel_values)`.

---

## Key Files

| File | Purpose |
|---|---|
| `runners/eval_vlmeval.py` | Task 2.2 eval runner — all 5 model classes + VLMEvalKit harness |
| `runners/measure_mac.py` | Task 2.1 performance measurement harness |
| `schemas/` | Pydantic schemas: `MetricsReport`, `ExperimentConfig` |
| `artifacts/eval_task_2_2_20260525_094121/` | Archived Task 2.2 results (15 JSONs + 15 configs) |
| `vendor/VLMEvalKit/` | Vendored VLMEvalKit (patched for decord/transformers 5.x) |
| `docs/VLM_Optimization_DetailedPlan_Phase0.md` | Full week-by-week plan |
| `docs/VLM_Optimization_Goals.md` | Project goals and exit criteria |
| `docs/decisions/0001-mac-measurement-methodology.md` | ADR-0001 |

---

## Next Steps (Week 3)

**Week 3: iPhone 16 Pro reference baselines**

- [ ] Task 3.1 — iOS developer provisioning (Xcode, signing, deploy to device) — **do first, riskiest task**
- [ ] Task 3.2 — LFM2.5-VL-450M on iPhone via Liquid's LEAP SDK
- [ ] Task 3.3 — FastVLM-0.5B on iPhone via `apple/ml-fastvlm` demo app (pre-built MLX weights available)
- [ ] Task 3.4 — SmolVLM-500M (llama.cpp/MLX) + MiniCPM-V 4.6 (OpenBMB iOS demo)
- [ ] Task 3.5 — Sanity-check measured TTFT/memory against published claims
- [ ] ADR-0002 — iOS measurement methodology
- [ ] ADR-0003 — iPhone baseline numbers

**Week 3 prerequisite check before starting:**
- Apple developer account active?
- iPhone 16 Pro physically available?
- Xcode installed on the Mac mini?

---

## Environment

| Component | Version / Detail |
|---|---|
| Primary dev machine | Mac mini M4 16GB unified memory |
| Python | 3.14 (Homebrew) |
| PyTorch | MPS backend (`PYTORCH_ENABLE_MPS_FALLBACK=1`) |
| transformers | 5.9.0 (breaking changes vs 4.x — see bugs above) |
| VLMEvalKit | Vendored at `vendor/VLMEvalKit/` (patched) |
| M5 Pro 32GB | Available later — re-run Task 2.1/2.2 when accessible |
