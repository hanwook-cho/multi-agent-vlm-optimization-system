# VLM Optimization — Phase 2 Detailed Plan

**Phase:** Phase 2 — Full System + Qwen2.5-VL-3B → Edge Model  
**Status:** Draft v1 (written 2026-06-08) — **CORRECTED 2026-06-13, see banner**  
**Planned duration:** 7–9 weeks  
**Starting point:** Phase 1 Pareto frontier; Qwen2.5-VL-3B as unoptimized teacher  
**Exit gate:** A ≤450M-class edge model derived from Qwen2.5-VL-3B running on iPhone 16 Pro (and Pi 5), competitive with at least two Phase 0 reference models, produced without manual configuration tweaks after setup

> **⚠️ CORRECTION (2026-06-13) — read before Strategy B/C below. See [ADR-0011](decisions/0011-phase2-strategy-correction.md).**
> This draft's **Strategy B distilled *into* LFM2-VL-450M / SmolVLM-500M as students — that is wrong.**
> Per Goals S3, LFM2-VL-450M is the **BENCHMARK** (the bar to match), not a student; the edge model's
> lineage must be **Qwen2.5-VL-3B**. Empirically, distilling into LFM2 **regressed** (POPE 86.2→38.5).
> Corrected approach (driven by the Search Strategist's hypotheses):
> - **P2-D2** — task-aligned distillation (teacher grounded-Q&A, not captions) + rehearsal. *Method validation only.*
> - **P2-B1** (primary) — assemble a right-sized open student (Qwen2.5-0.5B LM + small SigLIP) + distill from the 3B.
> - **P2-C1** — hard-prune the 3B (collapses toward P2-B1: vision 669M + embed 311M > 450M budget).
> The "Success targets" and "Strategy" sections below are superseded where they name LFM2/SmolVLM as students.

---

## Goal

Demonstrate the project's **central claim**: the system autonomously navigates the full optimization journey from a general-purpose 3B VLM (Qwen2.5-VL-3B, which cannot run on-device) down to a competitive 450M-class edge model on both target devices — in solo-developer calendar time, without manual config tweaking during the loop.

Phase 1 validated the loop works end-to-end. Phase 2 is the hard journey: not polishing an already-edge-optimized model, but *compressing a general-purpose 3B model into an edge-viable one*, crossing two orders of magnitude of model-size reduction while retaining quality competitive with the Phase 0 reference models.

This phase also builds the remaining infrastructure that Phase 1 deliberately deferred: the distillation pipeline, the Deployment Dispatcher, the human Approval Queue, and Pi 5 deployment support.

---

## What Phase 1 already built (carry-forward)

| Component | Status | Notes |
|---|---|---|
| `agents/search_strategist.py` | ✅ | Anthropic + Ollama backends, ReAct fallback |
| `services/experiment_runner.py` | ✅ | Mac quality proxy (fp16 MPS), writes ledger |
| `services/pareto_tracker.py` | ✅ | 4-axis dominance, phase-0 baselines hardcoded |
| `services/decision_dossier.py` | ✅ | ThresholdMonitor, scaffold; ready to fire |
| `ios_harness/` | ✅ | llama.cpp/mtmd Metal; `MeasurementSession.swift` |
| `schemas/experiments.py` | ✅ | ExperimentConfig, MetricsReport, n_ctx field |
| Stage A eval set | ✅ | 100 photos, 50 captions, 45 VQA pairs, hash-pinned |
| Pareto frontier | ✅ | H001 (LFM2 Q4_K_M CLIP 28.59), H002 (SmolVLM i1-Q4_0) |

---

## What Phase 2 needs to build

### New services

| Component | Location | Purpose |
|---|---|---|
| Caption distillation pipeline | `services/distillation_pipeline.py` | Teacher (Qwen2.5-VL-3B fp16) generates caption cache; student fine-tuning harness |
| Deployment dispatcher | `services/deployment_dispatcher.py` | Reads DeviceDescriptor, auto-selects GGUF conversion + quantization path, pushes artifacts to iOS harness and Pi 5 |
| Human approval queue | `services/approval_queue.py` | Append-only JSON log; CLI for human to approve/reject gated decisions |
| Fine-tune runner | `runners/finetune_vlm.py` | LoRA/QLoRA fine-tuning wrapper; writes to experiment ledger on completion |
| Pi 5 harness | `pi_harness/measure_pi.py` | llama.cpp GGUF inference + `llama_perf_context()` TPS counter (mirrors iOS harness) |

### Infrastructure changes

| Item | Description |
|---|---|
| Dashboard update | Add Qwen2.5-VL-3B "starting point" marker (quality-axis only, not yet edge-viable) and Phase 2 candidates to all plots |
| Pareto Tracker update | Add Pi 5 as second device axis; update dominance logic for per-device comparison |
| Search Strategist update | Extend hypothesis table to H-P2-001–H-P2-009; add `requires_training: bool` field to proposals so the Approval Queue knows which to gate |
| Stage B eval set (optional) | If personal-photo data arrives: add Stage B alongside Stage A. Does not block Phase 2. |

### Pi 5 unblock

Pi 5 deployment was deliberately skipped in Phase 0/1 (no hardware blocking). Phase 2 requires it for exit criterion 2.5. Dependencies:
- Pi 5 4 GB running Raspberry Pi OS (64-bit)
- llama.cpp compiled for `aarch64` with `LLAMA_BLAS=ON` (OpenBLAS) or `GGML_LLAMAFILE=OFF`
- Network-accessible from Mac (SSH + `rsync` for model push)
- `pi_harness/measure_pi.py` instrumented to emit `MetricsReport` JSON
- Risk: if Pi 5 hardware is unavailable, criterion 2.5 cannot be met. Document the gap; do not block the rest of Phase 2.

---

## Phase 2 hypothesis table

The hypotheses are grouped by strategy. The Search Strategist works through them in priority order, with Mac proxy evals before any iPhone or Pi dispatch.

### Strategy A — Pure quantization of Qwen2.5-VL-3B (no training)

| ID | Technique | Expected outcome | Priority |
|---|---|---|---|
| H-P2-001 | Qwen2.5-VL-3B Q4_K_M GGUF baseline (Mac) | Establish quality ceiling; confirm model converts cleanly | 1 — do first |
| H-P2-002 | Qwen2.5-VL-3B Q4_K_M iPhone run | TTFT ~50–90ms, Mem ~2.2–2.8GB; establish device feasibility | 2 — after H-P2-001 |
| H-P2-003 | Qwen2.5-VL-3B Q3_K_M | CLIP −2–4% vs Q4_K_M; on-disk ~1.2GB; still too large but probes the quality-size curve | 3 |
| H-P2-004 | Qwen2.5-VL-3B ctx-size 4096→512 | Mem −15%, TPS +5%; quality neutral; applies H005 learning | 4 |

**Decision gate after Strategy A:** If H-P2-002 shows Qwen2.5-VL-3B Q4_K_M runs on iPhone with TTFT < 80ms and Mem < 3GB, it may itself be the Phase 2 answer (no distillation needed). The Approval Queue checks this gate. If TTFT > 80ms or model doesn't fit on Pi 5 4GB, escalate to Strategy B.

### Strategy B — Knowledge distillation into existing 450M-class students

| ID | Technique | Student | Expected outcome | Priority |
|---|---|---|---|---|
| H-P2-005 | Caption cache distillation → LFM2-VL-450M | LFM2-VL-450M Q4_K_M | CLIP +1–3pts over H001; best chance of topping Phase 1 frontier | 1 |
| H-P2-006 | Caption cache distillation → SmolVLM-500M | SmolVLM-500M i1-Q4_0 | CLIP +2–4pts over H002; quality win on a fast model | 2 |
| H-P2-007 | Caption cache distillation → MiniCPM-V-4.6 | MiniCPM-V-4.6 Q4_K_M | Better MCQ benchmark scores; CLIP may improve less | 3 |

**Caption cache spec:** Qwen2.5-VL-3B fp16 on Mac M4 (or M5 Pro if available) generates open-ended descriptions for 50K images from COCO train2017. One overnight run (~8–12 hours). Cache stored as `datasets/caption_cache/qwen25_3b_coco50k.jsonl`. Student fine-tuning uses this cache only — teacher never runs again during training.

**Fine-tuning spec:** LoRA rank=16, α=32, target modules: `q_proj`, `v_proj`, `o_proj`. 3 epochs, lr=2e-4, cosine decay, batch=4 + gradient_accumulation=8 (effective batch 32). Mac M4 16GB: ~3–5 hours per student. 3 seeds per best candidate for statistical validity.

### Strategy C — Structural reduction of Qwen2.5-VL-3B

| ID | Technique | Expected outcome | Priority |
|---|---|---|---|
| H-P2-008 | LLM backbone swap: keep Qwen2.5-VL-3B vision encoder + smaller LLM backbone | Vision encoder quality + smaller text generation footprint | Contingency if B fails |
| H-P2-009 | Magnitude pruning Qwen2.5-VL-3B to 30% sparsity + fine-tune | Pruned 3B runs at ~2B-equivalent compute; quality depends on task distribution | Contingency if B fails |

Strategy C requires human-implemented Tier 2 code (new architecture swap, pruning harness). Only escalate here if Strategies A and B fail to produce a model within 10% of two reference models.

---

## Week-by-week task plan

### Week 1 — Qwen2.5-VL-3B baseline + infrastructure groundwork

**Goal:** Establish the "before" picture and unblock the main tools.

| Task | Description | Done when |
|---|---|---|
| P2-1.1 | Qwen2.5-VL-3B fp16 quality baseline on Mac (CLIP-score, n=50 Stage A) | MetricsReport written to ledger |
| P2-1.2 | GGUF conversion: `convert_hf_to_gguf.py` + `llama-quantize` Q4_K_M | `.gguf` file produced, loads in llama.cpp |
| P2-1.3 | Mac proxy quality eval on Q4_K_M GGUF (H-P2-001) | Ledger entry; CLIP-score vs fp16 delta measured |
| P2-1.4 | iPhone feasibility test (H-P2-002) | TTFT, TPS, Mem measured; iPhone verdict logged |
| P2-1.5 | Pi 5 setup (if hardware available) | Pi compiles llama.cpp; smoke test runs on `SmolVLM-500M.gguf` |
| P2-1.6 | `services/approval_queue.py` scaffold | `ApprovalQueue.push()` / `pop()` / `list_pending()` + CLI |

**Decision point end of Week 1:** Does Qwen2.5-VL-3B Q4_K_M run acceptably on-device? If TTFT < 80ms on iPhone AND fits on Pi 5 4GB: fast-track to P2 exit with just quantization variants (Weeks 2–3). If not: proceed to distillation (Weeks 3–7).

### Week 2 — Extended Mode A sweep on Qwen3B + Strategy A hypotheses

**Goal:** Squeeze everything possible from quantization alone before training.

| Task | Description | Done when |
|---|---|---|
| P2-2.1 | Q3_K_M variant (H-P2-003): Mac proxy + iPhone | Ledger entry; quality-size trade-off documented |
| P2-2.2 | ctx-size reduction (H-P2-004): Q4_K_M with n_ctx=512 | Memory reduction confirmed; no quality regression |
| P2-2.3 | Update Search Strategist hypothesis table with Phase 2 entries | Agent proposes Phase 2 hypotheses correctly |
| P2-2.4 | Pareto Tracker: add Pi 5 device axis | `ParetoTracker` handles multi-device frontier |
| P2-2.5 | Dashboard: Qwen2.5-VL-3B "starting point" marker on quality plot | Dashboard shows the before/after journey |
| P2-2.6 | `services/deployment_dispatcher.py` v1 | `dispatch(ExperimentConfig, DeviceDescriptor)` → produces `.gguf`, calls iOS harness; Pi path stubbed |

### Week 3 — Distillation pipeline build + caption cache generation start

**Goal:** Build the training infrastructure; start the overnight teacher run.

| Task | Description | Done when |
|---|---|---|
| P2-3.1 | `services/distillation_pipeline.py`: teacher caption generation | `generate_caption_cache(model_id, dataset, output_path)` writes `.jsonl` |
| P2-3.2 | Start caption cache generation: Qwen2.5-VL-3B fp16, 50K COCO train2017 images, run overnight | `datasets/caption_cache/qwen25_3b_coco50k.jsonl` written (10K sample verify, then full run) |
| P2-3.3 | `runners/finetune_vlm.py`: LoRA fine-tuning harness | `finetune(base_model_id, caption_cache_path, lora_config)` → `MetricsReport` on completion |
| P2-3.4 | Approval Queue wired to Search Strategist | Agent emits `requires_training: True` on training proposals; proposal paused pending human approve |
| P2-3.5 | Canary fine-tune test: 100-sample cache, 1 epoch, LFM2-VL-450M | Confirms training loop runs end-to-end; loss decreases |

**Compute estimate (caption cache generation):**
- Qwen2.5-VL-3B fp16 on Mac M4 16GB: ~500ms per image
- 50K images: ~7 hours wall-clock
- Memory peak: ~9GB (fp16 model + activations); M4 16GB runs it; M5 Pro 32GB preferred
- If M4 16GB cannot hold fp16: use `bfloat16` + MPS offload, or reduce to 20K images and scale up later

### Week 4 — Student fine-tuning runs (Strategy B)

**Goal:** Produce the first distilled candidates; measure on Mac.

| Task | Description | Done when |
|---|---|---|
| P2-4.1 | Fine-tune LFM2-VL-450M on caption cache (H-P2-005), 3 seeds | 3 Mac proxy MetricsReports; CLIP-score vs H001 delta |
| P2-4.2 | Fine-tune SmolVLM-500M on caption cache (H-P2-006), 3 seeds | 3 Mac proxy MetricsReports; CLIP-score vs H002 delta |
| P2-4.3 | Pareto update after fine-tune results | Frontier updated; best fine-tuned candidates identified |
| P2-4.4 | Decision Dossier check: is a fine-tuned model clearly better? | ThresholdMonitor confirms improvement or fires dossier |

**Compute estimate (fine-tuning):**
- LFM2-VL-450M LoRA on Mac M4 16GB: ~3 epochs × 50K samples ≈ 3–4 hours
- 3 seeds × 2 students = 6 training runs ≈ 24–30 hours total
- Runs overnight; no blocking

**Quality target:** The distilled LFM2-VL-450M should improve on the **MCQ benchmark set** (POPE/RealWorldQA/MMBench) toward the Qwen2.5-VL-3B teacher's scores, while not regressing CLIP-score below the Phase 0 baseline (27.6). See the revised Success Targets section — P2-1.1 showed the teacher is not a CLIP-score leader, so MCQ is the distillation signal.

### Week 5 — iPhone + Pi 5 evaluation of top candidates

**Goal:** Validate distilled models on both target devices.

| Task | Description | Done when |
|---|---|---|
| P2-5.1 | Quantize best fine-tuned LFM2-VL-450M to Q4_K_M GGUF | `.gguf` produced; Mac proxy confirms quality retained post-quant |
| P2-5.2 | iPhone run: fine-tuned LFM2-VL-450M Q4_K_M | TTFT, TPS, Mem measured; MetricsReport in ledger |
| P2-5.3 | iPhone run: fine-tuned SmolVLM-500M i1-Q4_0 (if competitive) | TTFT, TPS, Mem measured |
| P2-5.4 | Pi 5 run: top candidate (if hardware available) | TTFT, TPS, Mem on Pi 5 4GB; MetricsReport in ledger |
| P2-5.5 | MiniCPM-V-4.6 caption distillation (H-P2-007) if LFM2/SmolVLM results are below target | Only if needed per Pareto gap |

### Week 6 — Full benchmark evaluation + Phase 2 frontier

**Goal:** Complete the quality picture; confirm success criteria.

| Task | Description | Done when |
|---|---|---|
| P2-6.1 | VLMEvalKit run on best Phase 2 candidate: POPE, RealWorldQA, MMBench (n=100 each) | MacOS quality benchmark scores; compare to Phase 0 reference models |
| P2-6.2 | iPhone VLMEvalKit subset (n=50, same Stage A images) | On-device quality scores |
| P2-6.3 | Success criteria check: within 10% of ≥2 reference models? | Criteria 2.6 (S1, S2) pass/fail documented |
| P2-6.4 | If Strategy B insufficient: assess Strategy C options, file human-review dossier | Strategy C requires human implementation decision |
| P2-6.5 | Phase 2 Pareto frontier: at least 3 candidates per device with written rationale | Criterion 2.4 (P2) |
| P2-6.6 | Calendar time documentation: record Phase 0 start date → Phase 2 exit date | Criterion 2.7 (T1) |

### Week 7 — Documentation + arXiv preprint + blog post

**Goal:** Phase 2 public artifacts; close the phase.

| Task | Description | Done when |
|---|---|---|
| P2-7.1 | Reference model timeline comparison: SmolVLM v1→v2, LFM2→LFM2.5, FastViT→FastVLM | Chart: this project vs. reference teams' calendar time (T2) |
| P2-7.2 | Time-compression analysis: document hours of calendar time vs. median reference | Criterion 2.7 (T3); headline number for blog post |
| P2-7.3 | Phase 2 blog post: "The autonomous system produced a competitive edge VLM from a 3B general model" (~3500 words) | `docs/blog/phase2_reveal.md` written |
| P2-7.4 | arXiv preprint v1: system description + Phase 1+2 results | `.tex` + `.pdf` in `docs/arxiv/` |
| P2-7.5 | Demo video: model running on iPhone in real-time | Short screen recording |
| P2-7.6 | Experiment log published | All ledger entries and Pareto frontiers committed to public repo |

---

## Exit criteria (mapping to Goals doc §5)

| # | Goals criterion | Done when |
|---|---|---|
| 2.1 | All Phase 1 components + distillation pipeline, Deployment Dispatcher, Approval Queue, updated dashboard | All new services pass their integration tests |
| 2.2 | Caption cache distillation operational | Qwen2.5-VL-3B fp16 cache generated once; student training uses cache without re-running teacher |
| 2.3 | 450M-class model derived from Qwen2.5-VL-3B, produced without manual config tweaks (P1) | Search Strategist proposed the winning config autonomously; human approved the final deploy only |
| 2.4 | ≥3 candidates per device on the Pareto frontier with written rationale (P2) | Pareto Tracker shows ≥3 non-dominated points for iPhone; rationale in experiment ledger |
| 2.5 | Chosen model runs on iPhone 16 Pro and Pi 5 4 GB (P3) | MetricsReports from both devices in ledger; reproducible from committed artifacts |
| 2.6 | Chosen model within 10% of ≥2 reference models, on-disk ≤ reference at matched quantization (S1, S2) | VLMEvalKit scores confirm; on-disk size ≤ chosen reference's |
| 2.7 | Calendar time documented; reference-model timelines documented; time-compression delta is the headline (T1–T3) | Blog post and preprint both state the number explicitly |

---

## Approval gates (which decisions require human sign-off)

| Decision | Gate type | When |
|---|---|---|
| Start caption cache generation (overnight compute) | Explicit approval | Before P2-3.2 runs; human confirms ~7hr Mac compute |
| Start student fine-tuning (multi-day compute) | Explicit approval | Before P2-4.1/4.2; human confirms ~30hr training budget |
| Deploy to iPhone (first device run of each new candidate) | Explicit approval | Before P2-5.2, P2-5.3; per HLD §5.1 |
| Deploy to Pi 5 | Explicit approval | Before P2-5.4 |
| Escalate to Strategy C (structural reduction) | Decision Dossier | After Strategy B results measured; ThresholdMonitor fires if no Pareto improvement in 3 consecutive experiments |
| Approve fine-tuned candidate for public release | Explicit approval | Before P2-7.6 experiment log published |

All approvals logged to `artifacts/approval_queue.json` with timestamp and rationale.

---

## Success targets (quantitative)

> **Revised after P2-1.1 (2026-06-10).** The Qwen2.5-VL-3B teacher's CLIP-score
> (26.8 ± 3.7, n=5) is *not* above the edge models — CLIP-score of open-ended
> captions does not capture the teacher's advantage (verbosity + 77-token cap
> dilute it). The teacher's real edge is on **MCQ/reasoning benchmarks** (Phase 0:
> POPE 96.7, RealWorldQA 55, MMBench 66 — all measured). So the **distillation
> quality signal is the MCQ benchmark set, not CLIP-score.** CLIP-score is
> retained only as a no-regression sanity check. See
> [`docs/observations/2026-06-10-qwen25vl-3b-baseline.md`](observations/2026-06-10-qwen25vl-3b-baseline.md).

**Primary quality signal (distillation target — where the teacher leads):**

| Metric | Phase 0/1 best (edge) | Qwen2.5-VL-3B teacher | Phase 2 target |
|---|---|---|---|
| POPE % (Mac) | 91.7 (LFM2/MiniCPM) | **96.7** | ≥ 93, toward teacher |
| RealWorldQA % (Mac) | 42 (LFM2) | **55** | ≥ 48 |
| MMBench % (Mac) | 74 (LFM2) | 66 | ≥ 74 (no regression) |

**Sanity / no-regression checks (not the distillation target):**

| Metric | Phase 1 best | Phase 2 bound | Notes |
|---|---|---|---|
| CLIP-score | 28.59 (H001) | ≥ 27.6 (no regress vs Ph0 baseline) | Teacher CLIP ~26.8 — not a target to beat; just don't degrade caption alignment |
| TTFT ms (iPhone) | 15.2 (H001) | ≤ 20 | Slightly slower acceptable |
| TPS (iPhone) | 79.0 (H001) | ≥ 75 t/s | Small regression OK for quality gain |
| Peak Mem MB (iPhone) | 272 (H001) | ≤ 350 | Fine-tune doesn't change size |
| On-disk MB | 318 (H001) | ≤ 350 | Q4_K_M same size as H001 base |
| Pi 5 TPS | — | ≥ 5 t/s | Stretch (X2 from Goals doc) |

---

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Qwen2.5-VL-3B fp16 won't fit in Mac M4 16GB for caption generation | Medium | Blocks Strategy B | Use `bfloat16` + MPS offload; fall back to 20K sample cache; option to rent A100 ($50-100, see Q3) |
| Caption distillation doesn't improve CLIP over direct fine-tuning | Medium | Strategy B fails | Fall back to direct fine-tuning on Stage A + COCO captions (no teacher); lower quality gain but still meaningful |
| Pi 5 hardware unavailable | Low | Criterion 2.5 partial fail | Document gap; iPhone-only Phase 2 is still valid for the core claim; Pi 5 becomes Phase 3 debt |
| LoRA fine-tuning degrades TTFT/TPS on iPhone (larger weights from adapter merge) | Low | Pareto regression | Measure before/after adapter merge; if regression, use separate adapter file or reduce rank |
| Strategy B improves MCQ but regresses POPE below the 89% quality gate | Medium | Blocks deployment | Add POPE to LoRA training evaluation; check after every epoch; adjust training data balance |
| 3B → 450M jump exceeds what KD can bridge without structural change | Low | Strategy B ceiling | Strategy C (pruning or backbone swap) is the contingency; file Tier 2 hypothesis; human implements in Week 6 |
| Phase 2 calendar time exceeds 9 weeks | Medium | T3 criterion weakens | Document honestly; time compression is still the story if < 5 months total from Phase 0 start |

---

## Compute budget

| Item | Estimate | Notes |
|---|---|---|
| Caption cache generation (50K images) | ~7 hrs Mac M4 | One-time; runs overnight Week 3 |
| LoRA fine-tune × 6 runs (2 models × 3 seeds) | ~30 hrs Mac M4 | Split across Weeks 4-5; overnight |
| GGUF conversion + quantization | ~30 min per model | Negligible |
| VLMEvalKit full evaluation × 3 models | ~6 hrs Mac M4 | Week 6 |
| **Total Mac compute** | **~45 hrs** | Distributed across 7 weeks; no blocking waits |
| Cloud GPU option (if M4 OOMs on caption gen) | ~$50–100 A100 rental | Only if needed; budget pre-authorized per Goals Q3 |

---

## New hypothesis table format (Phase 2 extension)

```python
# Add to agents/search_strategist.py HYPOTHESIS_TABLE

PHASE2_HYPOTHESES = [
    {
        "id": "H-P2-001",
        "technique": "Qwen2.5-VL-3B Q4_K_M GGUF (Mac proxy)",
        "model": "Qwen2.5-VL-3B",
        "expected_gain": "CLIP > 30 (larger model quality ceiling)",
        "gain_axis": "quality",
        "status": "NOT_TRIED",
        "requires_training": False,
        "result_summary": "",
    },
    {
        "id": "H-P2-002",
        "technique": "Qwen2.5-VL-3B Q4_K_M GGUF (iPhone)",
        "model": "Qwen2.5-VL-3B",
        "expected_gain": "Device feasibility check — TTFT / Mem / TPS",
        "gain_axis": "latency+mem",
        "status": "NOT_TRIED",
        "requires_training": False,
        "result_summary": "",
    },
    {
        "id": "H-P2-005",
        "technique": "Caption cache distillation (Qwen3B → LFM2-VL-450M)",
        "model": "LFM2-VL-450M",
        "expected_gain": "CLIP +1–3 pts over H001 (28.59)",
        "gain_axis": "quality",
        "status": "NOT_TRIED",
        "requires_training": True,
        "result_summary": "",
    },
    # ... (H-P2-003, 004, 006-009 follow same structure)
]
```

The `requires_training: True` field flags proposals that need Approval Queue sign-off before execution. The Search Strategist is updated to include this field in `propose_experiment` output (new optional arg).

---

## Files to create in Phase 2

```
services/
  distillation_pipeline.py    ← caption cache generation + student training wrapper
  deployment_dispatcher.py    ← GGUF conversion, device push, iOS harness trigger
  approval_queue.py           ← append-only approval log + CLI

runners/
  finetune_vlm.py             ← LoRA/QLoRA fine-tuning; writes MetricsReport on exit

pi_harness/
  measure_pi.py               ← llama.cpp inference + MetricsReport export (mirrors iOS harness)
  requirements.txt            ← Pi-specific Python deps

datasets/
  caption_cache/
    .gitkeep                  ← directory committed; .jsonl files ignored (too large)

docs/
  VLM_Optimization_DetailedPlan_Phase2.md    ← this file
  blog/
    phase2_reveal.md          ← Phase 2 blog post (written at exit)
  arxiv/
    vlm_opt_system_v1.tex     ← arXiv preprint (written at exit)

tests/
  test_distillation_pipeline.py
  test_deployment_dispatcher.py
  test_approval_queue.py
```

---

## Phase 2 → Phase 3 transition

Phase 3 brings the Research Analyst Agent online (Mode B). The trigger for transition:

1. Phase 2 exit criteria all met, OR
2. Strategy B distillation has converged (ThresholdMonitor fires after 3 consecutive non-improvements and human chooses to escalate to Mode B rather than Strategy C)

At Phase 3 entry:
- The Phase 2 Pareto frontier is the baseline
- The Search Strategist resets its consecutive-non-improvement count (per HLD §10 D5)
- The Research Analyst Agent begins ingesting recent literature on efficient VLM techniques

---

*Companion documents: `docs/VLM_Optimization_Goals.md`, `docs/VLM_Optimization_HLD.md`, `docs/VLM_Optimization_DetailedPlan_Phase1.md`.*  
*Estimated calendar duration: 7–9 weeks solo at full-time-equivalent capacity.*
