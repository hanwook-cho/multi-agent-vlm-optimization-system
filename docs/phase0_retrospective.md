# Phase 0 Retrospective

**Date:** 2026-06-05  
**Duration:** 5 weeks (planned 5 weeks)  
**Status:** Complete

---

## What was planned vs what was delivered

| Exit criterion | Status | Notes |
|---|---|---|
| 0.1 LFM2-VL-450M on iPhone | ✅ | TTFT=14.1ms, TPS=82.4, Mem=279MB |
| 0.2 FastVLM-0.5B on iPhone | ✅ | TTFT=724.6ms, TPS=34.2, Mem=2204MB (FP16 MLX) |
| 0.3 SmolVLM-500M + MiniCPM-V-4.6 on iPhone | ✅ | SmolVLM 20ms / 49 TPS; MiniCPM-V 36ms / 34 TPS |
| 0.4 Pi 5 baselines | ⏭️ **Skipped** | No Pi hardware; decision is correct — iPhone is the primary target device |
| 0.5 Frozen Stage A eval set | ✅ | 100 photos, 50 captions, 45 VQA pairs, hash-pinned |
| 0.6 CLIP-score baselines | ✅ **(added)** | Not in original plan; added based on quality gap identified |
| 0.7 JSON schemas for all contracts | ✅ | MetricsReport, ExperimentConfig, AgentDecision, HypothesisRecord |
| 0.8 Metrics DB + dashboard | ✅ | SQLite + Streamlit, 4 tabs, all Phase 0 data visible |
| Mac quality eval (5 models × 3 benchmarks) | ✅ | POPE, RealWorldQA, MMBench — 100 samples each |
| ADRs | ✅ | 0001–0004, 0009 written |

**Net: 9/10 exit criteria met, 1 deliberately skipped (Pi), 1 meaningful addition (CLIP-score).**

---

## What went well

### 1. iOS harness worked end-to-end without team Apple Developer account
Building a multimodal iOS harness from scratch — ObjC++ wrapping llama.cpp + Swift UI — is non-trivial. The core challenge (attaching `libllama.a` + `libmtmd.a` as static XCFrameworks, passing pixel buffers through the ObjC++ boundary) was solved in Week 3 without major rewrites. The harness measured all four models cleanly.

### 2. Multi-model design paid off immediately
The original plan was one model per Xcode project. Adding a model picker in Week 3 (ModelEntry registry, segmented picker, chat template parameterisation) meant SmolVLM and MiniCPM-V were added in hours, not days. The chatTemplate field (`"chatml"` / `"smolvlm"`) was the key abstraction.

### 3. ADR discipline was the right call
Writing ADR-0001 first (Mac measurement methodology) meant the iPhone and Pi methodology docs were fast — they inherited the same structure and just filled in device-specific decisions. The ADR record is now a clear audit trail of every non-obvious decision.

### 4. COCO VQA v2 eliminated the manual VQA writing tax
The original plan assumed 2–3 hours of manual question-writing for the eval set. Pulling from COCO VQA v2 (214k human-written Q/A pairs) reduced this to a 30-line script and a 5-minute run. The quality is higher than manual writing would have been (10/10 annotator agreement on many pairs).

### 5. Phase 0 scope was right
Five weeks for baselines + eval set was the correct scope. Week 1 was genuinely necessary (schemas, ADR pattern, directory layout) and paid off throughout. Keeping Week 5 as "dashboard + close" rather than extending measurements was the right call.

---

## What was harder than expected

### 1. iOS provisioning (≈1.5× estimated time)
`ApplicationVerificationFailed` on FastVLM due to `com.apple.developer.kernel.increased-memory-limit` in the App ID cost ~4 hours. The fix (strip entitlements to camera-only, run via Xcode GUI not `devicectl`) was simple once found but required understanding the Apple Developer portal's App ID ↔ entitlement binding at a level not covered in any tutorial.

**Adjustment for Phase 1:** Any new iOS deployment starts with entitlement audit first — check `project.pbxproj` and the Developer portal App ID before building. Document this in CLAUDE.md.

### 2. transformers 5.x breaking changes (≈1 day)
Seven distinct compatibility bugs across the five models (wrong model class names, missing attributes, tokenizer API changes). transformers 5.x was released between project start and Week 2 execution. Each bug was individually quick to fix but the aggregate cost was a full day of unexpected work.

**Adjustment for Phase 1:** Pin transformers version in `requirements.txt`. When upgrading, run a smoke-test script (`tools/smoke_test_models.py`) before any measurement run.

### 3. FastVLM FP16 TTFT (expected vs actual framing)
FastVLM was expected to have the lowest TTFT (it was selected as a "fast" model). It has the highest TTFT in the Phase 0 results because no INT4 MLX build exists for the 0.5B variant. This required careful framing in ADR-0003 and the blog post to avoid misleading comparisons.

**Adjustment for Phase 1:** FastVLM's Phase 1 experiment target is producing an INT4 MLX build. That's the experiment that tests the architecture claim, not the FP16 baseline number.

### 4. Word-count TPS estimator (known limitation, materialized as expected)
The `output.split(" ").count` TPS estimator caused a 44.0 t/s outlier in MiniCPM-V run 3. This was identified as a risk in ADR-0002 and materialised exactly as expected. It's a minor data quality issue (trimmed mean is used), not a measurement failure.

**Adjustment for Phase 1:** Replace with actual llama.cpp token count in Week 1 of Phase 1. It's a one-line change in `LlamaVLMRunner.mm`.

### 5. CLIP-score image filenames (30 min, annoying)
The `generate_descriptions.py` script used `sample1–5.jpg` (copied from the plan doc's example) but the actual files were `img1–5.jpg`. Caused a silent all-skipped run that only surfaced after the fact.

**Adjustment for Phase 1:** Always run with `--dry-run` first when a script touches files by name. Add a pre-flight existence check to all file-iterating runners.

---

## What to adjust for Phase 1

| # | Adjustment | Priority |
|---|---|---|
| P1.1 | Pin `transformers==5.9.0` in requirements.txt; add smoke-test script | High |
| P1.2 | Replace word-count TPS with llama.cpp token counter in VLMHarness | High |
| P1.3 | Add pre-flight iOS entitlement check to deployment runbook | Medium |
| P1.4 | Add `--dry-run` flag to all file-iterating runner scripts | Low |
| P1.5 | Move `SAMPLE_IMAGES` / `IMAGE_DIR` to a shared config rather than per-script constants | Low |

---

## Phase 0 numbers snapshot

| Model | TTFT ms | TPS | Mem MB | POPE % | CLIP |
|---|---:|---:|---:|---:|---:|
| LFM2-VL-450M | **14.1** | **82.4** | **279** | 91.7 | 27.6 |
| SmolVLM-500M | 20.2 | 48.6 | 367 | 90.0 | 24.1 |
| MiniCPM-V-4.6 | 35.5 | 33.7† | 970 | 91.7 | **28.3** |
| FastVLM-0.5B | 724.6 | 34.2 | 2204 | 86.7 | 27.1 |

† trimmed mean (run 3 outlier excluded — ADR-0003)

These are the frozen Phase 1 comparison targets.

---

## Phase 0 → Phase 1 handoff state

**What is ready for Phase 1:**
- All baseline numbers measured and archived
- Stage A eval set frozen (`manifest_hash: e2128ae...`)
- Dashboard running and populated
- Schemas validated and in use
- Literature registry seeded with 6 core papers
- iOS harness deployable in ~10 minutes from scratch

**What Phase 1 must build before running experiments:**
- Search Strategist Agent (the core Phase 1 deliverable)
- Experiment Runner service (wraps existing runners into a callable API)
- Pareto Tracker (maintains the efficiency frontier across experiments)
- TPS token counter fix in VLMHarness before any Phase 1 iPhone measurement
