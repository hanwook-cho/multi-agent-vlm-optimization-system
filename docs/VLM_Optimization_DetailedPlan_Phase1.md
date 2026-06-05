# VLM Optimization — Phase 1 Detailed Plan

**Phase:** Phase 1 — Mode A Loop on Small-Model Compression  
**Status:** Draft v1 (written at Phase 0 closeout)  
**Planned duration:** 4–5 weeks  
**Starting point:** Phase 0 frozen baselines (LFM2-VL-450M on iPhone 16 Pro)  
**Exit gate:** Working closed-loop optimizer that has produced at least one Pareto improvement over Phase 0 baselines

---

## Goal

Build the **Search Strategist Agent** and its supporting services so that optimization experiments can be proposed, run, and evaluated autonomously. Phase 1 validates the loop works end-to-end on a tractable problem (compressing LFM2-VL-450M further) before Phase 2 attempts the harder Qwen2.5-VL-3B → 450M journey.

Phase 1 does NOT need to produce a state-of-the-art model. It needs to produce **one verifiable Pareto improvement** (e.g., 20% lower TTFT with ≤2pp POPE drop) and demonstrate that the loop can produce it autonomously.

---

## Exit criteria

| # | Criterion | Done when |
|---|---|---|
| 1.1 | Search Strategist Agent exists and runs | Agent proposes ≥3 valid experiment configs from Phase 0 data, dispatches them to runner, collects results |
| 1.2 | Experiment Runner service wraps existing runners | `runner.run(ExperimentConfig)` → `MetricsReport` without manual intervention |
| 1.3 | Pareto Tracker maintains frontier | After each run, tracker updates the (TTFT, quality) frontier and flags Pareto improvements |
| 1.4 | At least one Pareto improvement over Phase 0 baseline | LFM2-VL-450M variant with lower TTFT or memory at ≤2pp POPE/CLIP drop |
| 1.5 | Decision Dossier scaffold in place | Threshold Monitor defined, dossier format implemented, even if no escalation fires in Phase 1 |
| 1.6 | TPS token counter fixed in VLMHarness | Word-count estimator replaced with llama.cpp token count |
| 1.7 | Repository goes public | Repo visibility flipped, Phase 0+1 blog post published |

---

## Architecture

### Components to build

```
agents/
  search_strategist.py      ← NEW: core Phase 1 agent
  
services/
  experiment_runner.py      ← NEW: wraps runners/, dispatches to device
  pareto_tracker.py         ← NEW: maintains (TTFT, quality) frontier in DB
  evaluation_harness.py     ← NEW: runs Stage A eval set against a model
  
tools/
  smoke_test_models.py      ← NEW (retro P1.1): pre-flight model check
```

### Data flow

```
Search Strategist Agent
  ↓ reads Phase 0 MetricsReports + HypothesisRecords from DB
  ↓ proposes ExperimentConfig (technique, hyperparams, rationale)
  ↓ writes to experiments queue

Experiment Runner
  ↓ picks up ExperimentConfig
  ↓ applies technique (quantization, pruning, etc.) to model
  ↓ runs generate_descriptions.py + compute_clip_score.py on Stage A
  ↓ optionally deploys to iPhone + runs VLMHarness
  ↓ writes MetricsReport to DB

Pareto Tracker
  ↓ reads new MetricsReport
  ↓ compares to frontier (TTFT, memory, POPE, CLIP-score)
  ↓ flags Pareto improvements
  ↓ updates dashboard

Search Strategist Agent (next iteration)
  ↓ reads updated frontier + new results
  ↓ proposes next experiment
```

### Search Strategist Agent design

The agent is an LLM (Claude Sonnet) with:
- **Context:** Phase 0 baseline table, current Pareto frontier, HypothesisRecord history, literature registry excerpts
- **Tool:** `propose_experiment(technique, model, hyperparams, rationale)` → validates against ExperimentConfig schema, writes to queue
- **Policy:** Greedy-first on highest expected Pareto gain, avoid re-running already-explored configs (dedup via config hash), escalate to human when frontier hasn't improved in 3 consecutive experiments

The agent does NOT run experiments itself — it only proposes. The Experiment Runner is a separate process. This separation means the agent can't corrupt measurement data by hallucinating results.

---

## Week-by-week plan

### Week 1 — Infrastructure fixes + Experiment Runner

**Tasks:**

**1.1 — Fix VLMHarness TPS counter** *(P1 adjustment from retro, high priority)*  
Replace `output.split(" ").count` with actual llama.cpp decode token count via the existing callback. One-line change in `LlamaVLMRunner.mm` — validate with a 5-run LFM2 re-measurement. Update ADR-0003 with corrected TPS values if they differ materially.

**1.2 — Write `smoke_test_models.py`**  
A fast script that loads each model and runs one forward pass on a test image. Runs in < 5 minutes total. Catches transformers compatibility regressions before they waste a full measurement run.

**1.3 — Build `services/experiment_runner.py`**  
A Python class/function that accepts an `ExperimentConfig` and:
1. Applies the specified technique to the model (quantization → produces modified GGUF or HF checkpoint)
2. Runs `generate_descriptions.py` + `compute_clip_score.py` on Stage A (50-image caption set)
3. Optionally triggers iPhone measurement (manual step in Phase 1 — full automation is Phase 2)
4. Returns a `MetricsReport`

For Phase 1, "optionally trigger iPhone measurement" means the runner writes a ready flag and the human deploys to device using the existing VLMHarness. Full pipeline automation (SSH to device, pull results) is Phase 2.

**Done when:** `experiment_runner.py` can run a quantization experiment end-to-end from config to `MetricsReport`.

---

### Week 2 — Quantization experiments (AWQ INT4)

**Why first:** AWQ (2306.00978) is the highest-expected-gain experiment — INT4 quantization of the LM backbone reduces model size by ~50% and typically improves TPS by 30–50% with minimal accuracy drop (<2pp on most benchmarks). It's also the best-understood technique and most likely to produce a quick Pareto win.

**Tasks:**

**2.1 — AWQ INT4 quantization of LFM2-VL-450M LM backbone**  
Apply AWQ to the language model component using `llm-awq` or `autoawq`. Keep the vision encoder (mmproj) at its current Q8_0 — the LM backbone is the decode bottleneck, not the vision encoder.

Expected outcome: GGUF with INT4 LM + Q8_0 mmproj. Run via llama.cpp on iPhone.

Baseline to beat:
- TTFT: 14.1ms (should improve slightly — smaller KV cache, faster prefill)
- TPS: 82.4 t/s (should improve 20–40% — INT4 decode is faster than Q4_0 for this model size)
- Mem: 279MB (should decrease ~15%)
- POPE: 91.7% (must stay within ±2pp)
- CLIP-score: 27.6 (must not drop below 25.0)

**2.2 — SmolVLM-500M AWQ sweep** *(if 2.1 shows positive signal)*  
Same technique on the second model. Two data points establish whether AWQ generalises across the model family.

**Done when:** ≥1 AWQ result measured on iPhone, MetricsReport in DB.

---

### Week 3 — Vision encoder compression

**Why:** TTFT is dominated by vision-encoder prefill (ADR-0003, check 1). LFM2's mmproj at 99MB is already the smallest, but the prefill cost is still the first 12ms of the 14ms TTFT. Reducing the number of visual tokens passed to the LM would cut TTFT proportionally.

**Tasks:**

**3.1 — Visual token reduction experiment**  
LLaVA-style models produce one visual token per image patch. Reducing the patch count (e.g., by resizing the input image from 336px to 224px before the vision encoder) reduces visual token count by ~2.2× and should reduce prefill cost proportionally. This is a no-training change — just resize the input.

Risk: quality drop if the model was trained at 336px and lower-res inputs miss fine detail. Measure POPE and CLIP-score carefully.

**3.2 — Q4_0 mmproj experiment**  
The current mmproj is Q8_0. Quantising to Q4_0 halves its size (99MB → ~50MB). Expected impact: slight TTFT reduction and memory saving. Risk: small accuracy drop from lossy mmproj quantization.

**Done when:** ≥1 vision encoder experiment measured, result compared to Week 2 AWQ baseline.

---

### Week 4 — KV-cache compression + Pareto Tracker

**Tasks:**

**4.1 — KV-cache compression (SnapKV-style)**  
SnapKV (2404.14469) compresses the KV cache by identifying tokens that the model consistently attends to and evicting the rest during decoding. In the llama.cpp context, this can be approximated by setting a smaller `--ctx-size` and observing quality vs memory trade-off. A more principled implementation requires patching the llama.cpp attention layer.

Phase 1 target: measure the quality/memory trade-off at different `--ctx-size` values (512, 256, 128) to establish whether KV-cache compression is worth a deeper implementation in Phase 2.

**4.2 — Build `services/pareto_tracker.py`**  
Reads all `MetricsReport` records from DB. Maintains a Pareto frontier on configurable axes (default: TTFT vs POPE_acc). After each new result, identifies whether it dominates any existing point. Writes a `pareto_frontier.json` and updates the dashboard.

The Pareto Tracker is the key output consumers look at: "given everything we've tried, what's the best TTFT we've achieved at ≥90% POPE accuracy?"

**Done when:** Pareto Tracker runs after every new MetricsReport, frontier visible in dashboard.

---

### Week 5 — Search Strategist Agent + close

**Tasks:**

**5.1 — Build `agents/search_strategist.py`**  
The agent takes the current Pareto frontier + all previous HypothesisRecords as context and proposes the next experiment. Uses Claude API (claude-sonnet) with tool use.

Tools available to the agent:
- `propose_experiment(technique, model, hyperparams, rationale, expected_gain)` → validates config, writes to queue
- `query_literature(tag)` → returns relevant papers from `docs/literature/registry.json`
- `query_results(model_key)` → returns all MetricsReports for a model from DB

**Reasoning policy** (hardcoded in the system prompt):
1. Start from the technique with highest expected gain that hasn't been tried
2. If the last experiment was a Pareto improvement: explore a variation (hyperparameter sweep)
3. If the last experiment was not an improvement: try a different technique
4. After 3 consecutive non-improvements: flag for human review (don't escalate autonomously)

**5.2 — Decision Dossier scaffold**  
Define the `Threshold Monitor` — a simple check that fires when the Pareto frontier hasn't improved in N experiments. The dossier format is a markdown template that the agent fills in when the threshold fires. Phase 1 doesn't need the dossier to fire; the scaffold just needs to exist so Phase 2 can use it.

**5.3 — Phase 0+1 blog post + repo public reveal**  
Merge `docs/blog/draft_phase_0.md` content with Phase 1 results. Flip repo to public. Post on HN/Twitter.

**Done when:** Agent proposes at least one experiment autonomously from Phase 1 data (even if that experiment was already run manually).

---

## Experiment hypothesis table (seed)

Phase 1 starts with these hypotheses pre-registered in the HypothesisRecord table. The Search Strategist reads these as the initial known-techniques list.

| ID | Technique | Model | Expected gain | Risk | Paper |
|---|---|---|---|---|---|
| H001 | AWQ INT4 LM backbone | LFM2-VL-450M | TPS +30%, Mem -40% | POPE −2pp | 2306.00978 |
| H002 | AWQ INT4 LM backbone | SmolVLM-500M | TPS +25%, Mem -35% | POPE −2pp | 2306.00978 |
| H003 | Input resize 336→224px | LFM2-VL-450M | TTFT −30%, Mem −20% | CLIP −1.5 | architecture |
| H004 | Q4_0 mmproj (was Q8_0) | LFM2-VL-450M | Mem −12%, TTFT −5% | POPE −1pp | — |
| H005 | ctx-size 512 (was 2048) | LFM2-VL-450M | Mem −15%, TPS +5% | POPE −1pp | 2404.14469 |
| H006 | GPTQ INT4 LM backbone | LFM2-VL-450M | TPS +25%, Mem -40% | POPE −2pp | 2210.17323 |
| H007 | FastVLM INT4 MLX build | FastVLM-0.5B | TTFT −90%, Mem −65% | POPE −3pp | 2412.13303 |

H007 is the highest-impact experiment but also the riskiest (requires building INT4 MLX tooling for the FastVLM architecture). Defer to Week 3–4 pending AWQ results.

---

## Quality gates

Every experiment result is evaluated against these gates before being accepted as a Pareto candidate:

| Gate | Threshold | Rationale |
|---|---|---|
| POPE accuracy | ≥ 89.0% (Phase 0 min −2pp) | Hallucination must not regress substantially |
| CLIP-score | ≥ 25.0 (−2.5 from Phase 0 min) | Description quality floor |
| Peak memory | ≤ 3000 MB | iPhone 16 Pro practical limit without entitlement |
| On-device stability | No OOM crash on 5 test images | Stability requirement |

An experiment that passes all gates AND improves on ≥1 Pareto axis (TTFT, TPS, or memory) without worsening others is a Pareto improvement.

---

## What Phase 1 is NOT

- Phase 1 does **not** attempt Qwen2.5-VL-3B → 450M distillation. That is Phase 2.
- Phase 1 does **not** need a training pipeline. All Phase 1 experiments are post-training (quantization, input resize, ctx-size tuning).
- Phase 1 does **not** need fully automated iPhone deployment. Manual deploy with VLMHarness is acceptable. Automation is Phase 2.
- Phase 1 does **not** need Mode B (literature-driven research). The Search Strategist in Phase 1 only picks from the pre-registered hypothesis table.

---

## Phase 1 deliverables

| ID | Deliverable | Type |
|---|---|---|
| D1.1 | Working Search Strategist Agent code | Technical |
| D1.2 | Phase 1 Pareto frontier plot (Phase 0 baseline + Phase 1 best point) | Dashboard |
| D1.3 | Phase 0+1 combined reveal blog post | Public |
| D1.4 | Reproducible experiment bundle for the winning config | Technical |
| D1.5 | Search Strategist reasoning policy doc | Public |

---

## Key files to create in Phase 1

```
agents/
  search_strategist.py
  __init__.py

services/
  experiment_runner.py
  pareto_tracker.py
  evaluation_harness.py
  __init__.py

tools/
  smoke_test_models.py

ios_harness/VLMHarness/Inference/LlamaVLMRunner.mm   ← TPS fix
docs/blog/phase0_phase1_reveal.md                    ← merged reveal post
```
