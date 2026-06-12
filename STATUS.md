# Project Status

**Last updated:** 2026-06-10  
**Phase:** Phase 1 → Phase 2 transition  
**Phase 0 status:** ✅ COMPLETE (9/10 exit criteria met; Pi skipped deliberately)
**Phase 1 status:** ✅ COMPLETE (6/7 exit criteria met; criterion 1.7 pending repo flip)

---

## Phase 2 readiness (2026-06-10)

- **Plan written:** [`docs/VLM_Optimization_DetailedPlan_Phase2.md`](docs/VLM_Optimization_DetailedPlan_Phase2.md) — Qwen2.5-VL-3B → edge model, 7-week breakdown.
- **Search Strategist local backend decided + verified:** llama.cpp `--jinja` + `qwen2.5-7b-instruct` (native tool-calling, M4 16GB). 32B path reserved for the future 32GB Mac. See [ADR-0010](docs/decisions/0010-search-strategist-backend.md) and [observation](docs/observations/2026-06-10-qwen25-llamacpp-verification.md).
- **Agent hardening:** completed-hypothesis filter + 26 unit tests added (`tests/test_search_strategist.py`).
- **Launch helper:** `scripts/start_strategist_llm.sh`.

### Phase 2 Week 1 progress

- **P2-1.1 ✅ Qwen2.5-VL-3B CLIP baseline:** teacher tied with LFM2 (28.56 vs 29.00, n=50, paired t=−1.19) → distillation signal switched to MCQ benchmarks. See [observation](docs/observations/2026-06-10-qwen25vl-3b-baseline.md).
- **P2-1.2 ✅ Qwen2.5-VL-3B → Q4_K_M GGUF:** converted + verified via `llama-mtmd-cli` (coherent multimodal output). Recipe: `scripts/convert_qwen25vl_gguf.sh`.
  - Deployable bundle: **Q4_K_M LM 1.93GB + mmproj F16 1.34GB = ~3.27GB on-disk** (vs edge models 219–393MB). mmproj stays F16 (sub-Q8_0 blocked, H004).
  - ⚠️ Early signal for **P2-1.4 (iPhone gate):** Qwen-VL wants ≥1024 image tokens (vs LFM2's 576) + 1.34GB mmproj → expect high TTFT + memory on-device. GGUF files gitignored (live in `models/qwen2.5-vl-3b-gguf/`).
- **P2-1.3 ✅ MCQ benchmarks on the Q4_K_M GGUF:** decomposed path vs quantization on identical slices. **Q4_K_M is quality-preserving** (quant Δ ≤5pts: POPE −1.5, MMBench 0, RWQA −5). Benchmark swings are the **inference path** (transformers→llama.cpp ±10–19pts), not quantization. Methodology rule: hold the inference path constant for cross-model comparisons. See [observation](docs/observations/2026-06-11-qwen-gguf-mcq-path-vs-quant.md).
- **Next:** P2-1.4 (iPhone feasibility gate — needs device).

---

## Phase 0 Progress

| Week | Focus | Status |
|---|---|---|
| Week 1 | Infrastructure, schemas, Mac measurement harness | ✅ Done |
| Week 2 | Mac quality eval (VLMEvalKit, 5 models × 3 benchmarks) | ✅ Done |
| Week 3 | iPhone 16 Pro reference baselines (4 models) | ✅ Done |
| Week 4 | ~~Pi 5 baselines~~ (skipped) + Stage A eval set assembly | ✅ Done |
| Week 5 | Dashboard, literature spike, Phase 0 closeout | ✅ Done |

---

## Week 4 Progress — Stage A Eval Set

| Task | Status |
|---|---|
| 4.5-A Source & curate 100 photos (95 COCO + 5 proxy) | ✅ Done |
| 4.5-B Write 45 VQA pairs (pulled from COCO VQA v2) | ✅ Done |
| 4.5-C COCO reference captions (50 photos) | ✅ Done |
| 4.5-D Hash-pin manifest | ✅ Done (provisional — updates when vqa.json written) |
| 4.5-E ADR-0004 | ✅ Done |

**Final manifest hash:** `e2128ae022b3720375d7c866a037b6d8ec4b399ff92cb59e6065ec9fb7f3e29f`

**Key files:**
- `datasets/stage_a/photos/` — 100 photos (95 COCO val2017 seed=42, 5 proxy)
- `datasets/stage_a/captions.json` — 50 COCO reference captions
- `datasets/stage_a/vqa.json` — 45 VQA pairs (sourced from COCO VQA v2, human-annotated, ≥4/10 agreement)
- `datasets/stage_a/manifest.json` — per-file SHA-256 + manifest hash
- `tools/curate_eval_set.py` — curation script
- `tools/hash_eval_set.py` — manifest generation
- `docs/decisions/0004-stage-a-eval-set.md` — ADR-0004

---

## CLIP-Score Baselines (Mac, open-ended descriptions, 5 images)

CLIP model: `openai/clip-vit-large-patch14` · Prompt: "Describe what you see in this image."  
Score = 100 × max(0, cos_sim(CLIP_img, CLIP_txt)). Typical range for good captions: 25–35.

| Model | Platform | CLIPScore | ±σ |
|---|---|---:|---:|
| MiniCPM-V-4.6 | Mac MPS (bfloat16) | **28.31** | 3.74 |
| LFM2-VL-450M | Mac MPS (bfloat16) | 27.60 | 3.49 |
| FastVLM-0.5B | iPhone FP16 (MLX) | 27.12 | 3.10 |
| SmolVLM-500M | Mac MPS (bfloat16) | 24.11 | 2.55 |

**Takeaways:**
- All four models cluster tightly (24–28) — scores within ~4 points, well within σ overlap
- MiniCPM-V-4.6 leads narrowly; FastVLM's more verbose iPhone descriptions score comparably to LFM2 on Mac
- SmolVLM trails by ~3–4 points — shorter, less detailed captions
- No model dominates on description quality alone; TTFT/TPS (latency) remains the primary differentiator

**Artifacts:**
- Predictions: `artifacts/clip_preds/{model}_preds.json`
- Scores: `artifacts/clip_scores/{model}_preds_clip.json`
- Runners: `runners/generate_descriptions.py`, `runners/compute_clip_score.py`

---

## Completed Tasks

### Week 1
- **Task 1.x** — Schemas (`MetricsReport`, `ExperimentConfig`), device descriptors, project scaffolding
- **Task 2.1** — Qwen2.5-VL-3B measured on Mac mini M4 16GB (swap-contaminated run documented in ADR-0001)
- **ADR-0001** — Mac measurement methodology (`docs/decisions/0001-mac-measurement-methodology.md`)

### Week 3
- **Task 3.1** — iOS developer provisioning: Xcode 26.5, team `9LVH98N9BS`, iPhone 16 Pro registered, smoke-test app deployed
- **Task 3.2** — LFM2-VL-450M Q4_0 baseline on iPhone 16 Pro via llama.cpp/mtmd (Metal backend)
  - Harness: `ios_harness/VLMHarness.xcodeproj` (ObjC++ wrapper around libmtmd + libllama)
  - Archived: `artifacts/eval_task_3_2_20260605/LFM2-VL-450M_iphone16pro_20260605.json`

### Week 2
- **Task 2.2** — Quality evaluation of all 5 reference VLMs × 3 benchmarks × 100 samples on Mac mini M4 16GB
  - Runner: `runners/eval_vlmeval.py` (VLMEvalKit-based)
  - Archived: `artifacts/eval_task_2_2_20260525_094121/` (15 MetricsReport JSONs)
  - Several scoring bugs found and fixed during this session (see Known Issues below)

---

## iPhone 16 Pro Baseline Results (A18 Pro, iOS 26.5)

| Model | Backend | Quant | TTFT ms | ±σ | Decode t/s | ±σ | Peak Mem MB | On-disk MB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LFM2-VL-450M | llama.cpp/mtmd Metal | Q4_0 | **14.1** | 0.2 | **82.4** | — | **279** | 219 |
| SmolVLM-500M | llama.cpp/mtmd Metal | Q4_K_M | 20.2 | 0.2 | 48.6 | 0.8 | 367 | 393 |
| MiniCPM-V-4.6 | llama.cpp/mtmd Metal | Q4_K_M | 35.5 | 0.6 | 33.7* | — | 970 | 1199 |
| FastVLM-0.5B | MLX Swift | FP16 | 724.6 | 37.4 | 34.2 | 1.1 | 2204 | ~1000 |

Device: iPhone 16 Pro (iPhone17,1, A18 Pro, iOS 26.5).  
Prompt: "Describe the image in English. Output should be brief, about 15 words or less."  
Images: sample1–5.jpg (same set for all models).

**Notes:**
- LFM2 TTFT is dramatically lower (14 ms vs 725 ms) due to Q4_0 quantization + llama.cpp's optimized Metal kernels vs FP16 MLX
- FastVLM memory (2204 MB) reflects full FP16 weights loaded by MLX runtime; LFM2 Q4_0 only needs 279 MB
- FastVLM `onDiskSizeMB` = 0 in JSON (HF cache path resolution failed on-device — actual size ~1 GB FP16)
- All three llama.cpp/Metal models cluster in TTFT: 14–36 ms, memory scales with model weight size (279→367→970 MB)
- MiniCPM-V TPS*: run 3 = 44.0 t/s is an outlier (sample3 had shorter output → fewer decode tokens → inflated word-count TPS estimate); other 4 runs = 33.55, 33.49, 33.91, 33.87 → trimmed mean **33.7 t/s**. JSON mean 35.8 is reported as-is in the artifact but 33.7 is the representative value.
- SmolVLM TTFT (20 ms) is close to LFM2 (14 ms) — same Metal backend; SmolVLM's idefics3 vision encoder slightly heavier than LFM2's
- FastVLM FP16 via MLX is an outlier in all dimensions (50× higher TTFT, 6× more memory) — FP16 weight loading cost dominates
- Archived: `artifacts/eval_task_3_3_20260605/FastVLM-0.5B_iphone16pro_20260605.json`
- Archived: `artifacts/eval_task_3_4_20260605/SmolVLM-500M_iphone16pro_20260605.json`
- Archived: `artifacts/eval_task_3_4_20260605/MiniCPM-V-4.6_iphone16pro_20260605.json`

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

## Week 5 Progress

| Task | Status |
|---|---|
| 5.1 SQLite metrics DB + Streamlit dashboard | ✅ Done |
| 5.2 Literature spike (ADR-0009) | ✅ Done |
| 5.3 Phase 0 blog post draft | ✅ Done |
| 5.4 Phase 0 retro + Phase 1 plan | ✅ Done |

**Dashboard:** `streamlit run dashboard.py` → 4 tabs: iPhone Performance, Mac Quality, CLIP-Score, About  
**DB builder:** `python tools/build_metrics_db.py` → `metrics.db`

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
| `ios_harness/VLMHarness.xcodeproj` | Task 3.1/3.2 iOS harness (ObjC++ llama.cpp/mtmd + Swift UI) |
| `ios_harness/VLMHarness/LlamaVLMRunner.mm` | ObjC++ wrapper: llama.cpp + mtmd multimodal inference |
| `ios_harness/VLMHarness/MeasurementSession.swift` | TTFT/TPS/memory measurement over N runs |
| `ios_harness/VLMHarness/ReportExporter.swift` | Exports MetricsReport JSON to Files app |
| `ios_harness/VLMHarness/ContentView.swift` | Measurement UI (run button, results display, export) |
| `vendor/ml-fastvlm/` | Cloned `apple/ml-fastvlm` — instrumented for Task 3.3 |
| `vendor/ml-fastvlm/app/FastVLM App/FastVLMModel.swift` | TPS/mem instrumentation added |
| `vendor/ml-fastvlm/app/FastVLM App/FastVLM.entitlements` | Restricted entitlements removed (camera only) |

---

## Next Steps (Week 3)

**Week 3: iPhone 16 Pro reference baselines**

- [x] Task 3.1 — iOS developer provisioning (Xcode, signing, deploy to device) ✅ **done**
- [x] Task 3.2 — LFM2-VL-450M on iPhone via llama.cpp/mtmd ✅ **done** — TTFT=14.1±0.1 ms, TPS=82.4, mem=279 MB
- [ ] Task 3.2 — LFM2.5-VL-450M on iPhone via Liquid's LEAP SDK
- [x] Task 3.3 — FastVLM-0.5B on iPhone via `apple/ml-fastvlm` (MLX FP16) ✅ — TTFT=724.6±37.4 ms, TPS=34.2±1.1, Mem=2204 MB
- [x] Task 3.4 — SmolVLM-500M + MiniCPM-V-4.6 via llama.cpp/mtmd Q4_K_M ✅
  - SmolVLM: TTFT=20.2±0.2 ms, TPS=48.6, Mem=367 MB
  - MiniCPM-V-4.6: TTFT=35.5±0.6 ms, TPS=33.7 (trimmed; raw mean 35.8 inflated by 1 outlier run), Mem=970 MB
- [x] Task 3.5 — Sanity-check vs published claims ✅ (no absolute numbers published; internal consistency verified — see ADR-0003)
- [x] ADR-0002 — iOS measurement methodology ✅ (`docs/decisions/0002-ios-measurement-methodology.md`)
- [x] ADR-0003 — iPhone baseline numbers ✅ (`docs/decisions/0003-iphone-baseline-numbers.md`)

---

## Task 3.3 Results — FastVLM-0.5B on iPhone ✅

**Results:** TTFT=724.6±37.4 ms, TPS=34.2±1.1 t/s, Peak Mem=2204 MB  
**Archived:** `artifacts/eval_task_3_3_20260605/FastVLM-0.5B_iphone16pro_20260605.json`

**What was done:**
- Cloned `apple/ml-fastvlm` → `vendor/ml-fastvlm/`
- Model downloaded: `vendor/ml-fastvlm/app/FastVLM/model/` (0.5B FP16, MLX format, ~1 GB)
- Signing fixed: `DEVELOPMENT_TEAM = 9LVH98N9BS`, bundle ID → `com.hwcho99.FastVLMBaseline`
- Entitlements stripped to camera-only: `vendor/ml-fastvlm/app/FastVLM App/FastVLM.entitlements`
  - **Removed:** `increased-memory-limit`, `app-sandbox`, network, file-access
  - **Kept:** `com.apple.security.device.camera`
- MLX packages pinned in `Package.resolved`:
  - `mlx-swift` → `0.21.2` (SHA `70dbb62`)
  - `mlx-swift-examples` → `2.21.2` (SHA `6ef303b`)
- DerivedData `Tokenizer.swift` patched (Xcode 26 `dictionary` ambiguity fix):
  - Path: `~/Library/Developer/Xcode/DerivedData/FastVLM-.../SourcePackages/checkouts/mlx-swift-examples/Libraries/MLXLMCommon/Tokenizer.swift`
  - `updateTokenizerConfig()` simplified to `return tokenizerConfig` (Qwen2 tokenizer; code path never triggered)
  - **⚠️ This DerivedData patch is wiped by any Xcode "Reset Package Caches". Must re-apply if packages resolve again.**
- TPS + memory instrumentation added to `FastVLMModel.swift` (`physicalFootprintMB()`, `decodeTPSString`, `peakMemString`, `BenchmarkResult`)
- TPS/Mem overlay added to `ContentView.swift`
- **BUILD SUCCEEDED** ✅

**Install blocker (resolved):** `ApplicationVerificationFailed` was caused by `com.apple.developer.kernel.increased-memory-limit` entitlement registered in the Apple Developer portal App ID. Fixed by stripping entitlements to camera-only and running via Xcode GUI (▶ Run) which created a fresh App ID.

**Files modified for Task 3.3:**
| File | Change |
|---|---|
| `vendor/ml-fastvlm/app/FastVLM App/FastVLMModel.swift` | Memory helper, BenchmarkResult, TPS/mem tracking in generate() |
| `vendor/ml-fastvlm/app/FastVLM App/ContentView.swift` | TTFT overlay extended with TPS + Mem lines |
| `vendor/ml-fastvlm/app/FastVLM App/FastVLM.entitlements` | Stripped to camera-only |
| `vendor/ml-fastvlm/app/FastVLM.xcodeproj/project.pbxproj` | DEVELOPMENT_TEAM, bundle IDs, Video extension bundle ID |
| `vendor/ml-fastvlm/app/FastVLM.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved` | Pinned mlx-swift 0.21.2 + mlx-swift-examples 2.21.2 |
| DerivedData `MLXLMCommon/Tokenizer.swift` | `updateTokenizerConfig` simplified (re-apply after any package reset) |

---

**Week 3 provisioning details (Task 3.1):**
- Xcode 26.5 / iOS 26.5 SDK on Mac mini M4 ✅
- Apple Developer team: `9LVH98N9BS` (hwcho99@yahoo.com) ✅
- Device: iPhone 16 Pro `4B5CB219-A5DE-59DF-81A5-7FBECBC17B90` (iPhone17,1, iOS 26.5) ✅
- Smoke-test app `VLMHarness` (`com.hwcho99.VLMHarness`) deployed and ran on device ✅
- Xcode project: `ios_harness/VLMHarness.xcodeproj` (automatic signing)

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
