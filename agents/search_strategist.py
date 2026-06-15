"""
agents/search_strategist.py
────────────────────────────
Search Strategist Agent for Phase 1 VLM optimization.

The agent reads the current Pareto frontier and all previous experiment
results, then calls the Claude API with tool use to propose the next
experiment to run.

Tools available to the agent:
  - propose_experiment   → validates ExperimentConfig, writes to queue
  - query_results        → returns MetricsReports for a model from ledger
  - query_frontier       → returns current Pareto frontier

Reasoning policy (hardcoded in system prompt):
  1. Start from the technique with highest expected gain that hasn't been tried
  2. If last experiment was a Pareto improvement → explore a variation
  3. If last experiment was not an improvement → try a different technique
  4. After 3 consecutive non-improvements → flag for human review

Usage:
    from agents.search_strategist import SearchStrategist

    agent = SearchStrategist()
    proposal = agent.propose_next()
    print(proposal)          # ExperimentProposal with rationale + config
    agent.print_report()     # full reasoning trace
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import anthropic as _anthropic_mod
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import openai as _openai_mod
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# ── Project root ──────────────────────────────────────────────────────────────

_HERE        = Path(__file__).parent
PROJECT_ROOT = _HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.experiments import ExperimentConfig, CompressionSpec
from schemas.students import StudentSpec
from services.pareto_tracker import ParetoTracker, PHASE0_BASELINES, DEFAULT_AXES

LEDGER_DIR  = PROJECT_ROOT / "artifacts" / "experiment_ledger"
QUEUE_FILE  = PROJECT_ROOT / "artifacts" / "experiment_queue.json"
# Tier-2 student-construction proposals (P2-B1, ADR-0012) queue here; the
# construction loop (services/construction_loop.py) consumes and builds them.
CONSTRUCTION_QUEUE = PROJECT_ROOT / "artifacts" / "construction_queue.json"

STAGE_A_HASH = "e2128ae022b3720375d7c866a037b6d8ec4b399ff92cb59e6065ec9fb7f3e29f"

# ── Local backend defaults ──────────────────────────────────────────────────
# Primary local backend is llama.cpp's server with Qwen2.5-7B-Instruct.
# Qwen2.5 has native tool-calling, so with `--jinja` the OpenAI `tools` API
# works directly (no ReAct fallback needed). Launch with:
#   llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf --jinja --port 8080 -c 8192
# See scripts/start_strategist_llm.sh.
#
# Hardware note: Qwen2.5-7B Q4_K_M (~5GB runtime) is the default for the
# M4 16GB Mac mini. On a 32GB machine, step up to qwen2.5-32b-instruct
# (set LLAMACPP_MODEL + serve the 32B GGUF). See ADR-0010.
LLAMACPP_BASE_URL = "http://localhost:8080/v1"
LLAMACPP_MODEL    = "qwen2.5-7b-instruct"  # cosmetic — llama-server uses the loaded GGUF

# Operator-facing backend aliases (ADR-0013): the operator thinks "local" vs "api".
_BACKEND_ALIASES = {"local": "llamacpp", "api": "anthropic"}


def _resolve_backend_name(backend: str = "auto", env: dict | None = None) -> str:
    """Resolve the backend name. DEFAULT IS LOCAL (llama.cpp + Qwen2.5).

    API backends are strictly OPT-IN — set STRATEGIST_BACKEND=api (or 'anthropic'),
    or pass backend= explicitly. The mere presence of ANTHROPIC_API_KEY does NOT
    switch to the API: runs stay local (private + free) unless the operator opts in
    (ADR-0013 — chat/agent backend configurable, default local).
    """
    if backend and backend != "auto":
        return _BACKEND_ALIASES.get(backend, backend)
    env = env if env is not None else os.environ
    choice = env.get("STRATEGIST_BACKEND", "").strip().lower()
    if choice:
        return _BACKEND_ALIASES.get(choice, choice)
    return "llamacpp"

# ── Hypothesis table (seed — matches Phase 1 plan) ───────────────────────────

HYPOTHESIS_TABLE = [
    {
        "id": "H001",
        "technique": "k-quant imatrix (Q4_K_M)",
        "model": "LFM2-VL-450M",
        "expected_gain": "CLIP +1–2 (better weight calibration than Q4_0)",
        "gain_axis": "quality",
        "status": "CONFIRMED",
        "experiment_id": "12d065239be5693a9e3aa57bcc6e0a814143c00145de441fc29d17ad5922d580",
        "result_summary": "CLIP 28.59 (+3.6% vs Q4_0 baseline). TTFT 15.2ms, TPS 78.9, Mem 272MB.",
    },
    {
        "id": "H002",
        "technique": "imatrix Q4_0 (i1-Q4_0)",
        "model": "SmolVLM-500M",
        "expected_gain": "TPS +5–15%, Mem -5%, quality neutral",
        "gain_axis": "speed+mem",
        "status": "CONFIRMED",
        "experiment_id": "ccd9d9bca7d6c15ff1d9fa7196fa9f57d412a437d2a052f5b79c25a6c9d9a30e",
        "result_summary": "CLIP 27.78 vs Ph0 24.11 (fp16 proxy diff). TPS 51.9 (+7%), TTFT 17.7ms (-12%).",
    },
    {
        "id": "H003",
        "technique": "Input resize 336→224px",
        "model": "LFM2-VL-450M",
        "expected_gain": "TTFT -30%, Mem -20% (fewer visual tokens)",
        "gain_axis": "latency",
        "status": "NULL_RESULT",
        "experiment_id": "a8d879818a188ad87d29e60a67e17af77b6994910e47b1ff1af4ea2987e63dce",
        "result_summary": (
            "TTFT 15.43ms (+1.2% vs H001 — no change). Root cause: llama.cpp CLIP "
            "preprocessor baked into mmproj always resizes to model-native 336px, "
            "overriding the upstream downscale. Technique does not work on GGUF path."
        ),
    },
    {
        "id": "H004",
        "technique": "Q4_0 mmproj (was Q8_0)",
        "model": "LFM2-VL-450M",
        "expected_gain": "Mem -12%, TTFT -5%",
        "gain_axis": "mem+latency",
        "status": "BLOCKED",
        "result_summary": (
            "llama-quantize rejects CLIP arch. gguf-py OOM (1.2GB free). "
            "convert_hf_to_gguf.py only supports up to q8_0. All quantization "
            "paths for mmproj are blocked with current tooling."
        ),
    },
    {
        "id": "H005",
        "technique": "ctx-size reduction (4096→1024)",
        "model": "LFM2-VL-450M",
        "expected_gain": "Mem -15%, TPS +5%",
        "gain_axis": "mem",
        "status": "DEFERRED",
        "phase": 1,
        "result_summary": "Phase 1 quantization tweak — deferred; Phase 1 is complete and the project is now on the Phase 2 (3B→edge) goal.",
    },
    {
        "id": "H006",
        "technique": "GPTQ INT4 LM backbone",
        "model": "LFM2-VL-450M",
        "expected_gain": "CLIP +0–1, POPE neutral (quality-preserving INT4)",
        "gain_axis": "quality",
        "status": "DEFERRED",
        "phase": 1,
        "result_summary": "Phase 1 quantization tweak — deferred; project is now on the Phase 2 (3B→edge) goal.",
    },
    {
        "id": "H007",
        "technique": "FastVLM INT4 MLX build",
        "model": "FastVLM-0.5B",
        "expected_gain": "TTFT -90%, Mem -65% (vs fp16 baseline)",
        "gain_axis": "latency+mem",
        "status": "DEFERRED",
        "phase": 1,
        "result_summary": "Phase 1 quantization tweak — deferred; project is now on the Phase 2 (3B→edge) goal.",
    },
    # ── Phase 2: produce an edge model FROM Qwen2.5-VL-3B (open, general),
    #    competitive with the LFM2-VL-450M BENCHMARK. LFM2 is the target/yardstick,
    #    not a student. All Phase 2 techniques are Tier-2 (code, human-implemented).
    {
        "id": "P2-D1",
        "technique": "Caption-only distillation (Qwen2.5-VL-3B teacher → student)",
        "model": "LFM2-VL-450M (student)",
        "expected_gain": "lift student MCQ toward teacher",
        "gain_axis": "quality",
        "status": "REGRESSED",
        "phase": 2,
        "tier": "code",
        "result_summary": (
            "5K Qwen captions LoRA-distilled into LFM2-VL-450M. REGRESSED on every MCQ "
            "(same-path): POPE 86.2→38.5, RWQA 42→36, MMBench 74→57. Caption-only objective "
            "caused task interference / forgetting of grounding (answers well-formed but wrong). "
            "Two lessons: (a) distill the skill we MEASURE (grounded Q&A), not captions; "
            "(b) LFM2 is the BENCHMARK, not a valid student — it is already edge-optimized."
        ),
    },
    {
        "id": "P2-D2",
        "technique": "Task-aligned distillation (teacher answers grounded VQA/yes-no/MCQ → student) + rehearsal",
        "model": "LFM2-VL-450M (method-validation pilot base)",
        "expected_gain": "lift POPE/RWQA/MMBench toward teacher without forgetting",
        "gain_axis": "quality",
        "status": "REGRESSED",
        "phase": 2,
        "tier": "code",
        "result_summary": (
            "11.2K task-aligned teacher Q&A (yes-no + open) + 20% caption rehearsal, LoRA into "
            "LFM2-VL-450M, 3 epochs. STILL REGRESSED on every MCQ (same-path, n=100): "
            "POPE 87.7→66.7, RWQA 42→37, MMBench 74→51. The POPE failure mode FLIPPED vs P2-D1: "
            "student now answers 'Yes' to ~every presence question (acc 50/prec 50/recall 100 = "
            "always-yes collapse), because the teacher Q&A asked mostly about objects that ARE "
            "present → presence-bias prior. Two lessons: (a) naive teacher-generated Q&A is "
            "data-imbalanced — needs balanced hard negatives (absent-object questions) before it "
            "can teach grounding; (b) the whole D-series (distill INTO LFM2) is exhausted as a way "
            "to BEAT the benchmark — LFM2 is already edge-optimized, so any LoRA perturbation only "
            "moves it off its tuned optimum. Pivot to building a RIGHT-SIZED student from the 3B "
            "(P2-B1), where distillation adds capability instead of overwriting it."
        ),
    },
    {
        "id": "P2-C1",
        "technique": "Structured prune Qwen2.5-VL-3B → ~450M + distill-recover",
        "model": "Qwen2.5-VL-3B (compressed)",
        "expected_gain": "edge-size model with direct 3B lineage",
        "gain_axis": "size",
        "status": "NOT_TRIED",
        "phase": 2,
        "tier": "code",
        "result_summary": "Architecture constraint: vision encoder 669M + embeddings 311M = 980M ALONE > 2× the 450M target. Pure LM-layer pruning cannot reach the budget; must also replace the vision encoder and shrink the vocab — collapses toward an assemble-small-student rebuild (P2-B1).",
    },
    {
        "id": "P2-B1",
        "technique": "Assemble small student (Qwen2.5-0.5B LM + small SigLIP vision) + distill from 3B",
        "model": "assembled ~450–600M VLM",
        "expected_gain": "edge-size model matching the LFM2 benchmark; right-sized architecture from the start",
        "gain_axis": "size+quality",
        "status": "IN_PROGRESS",
        "phase": 2,
        "tier": "code",
        "result_summary": (
            "FIRST constructed student built end-to-end by the system (ADR-0012 B1.3, spec df64c49b): "
            "Qwen2.5-0.5B + SigLIP-base, fresh MLP projector, align 200 + distill 1000 steps on the "
            "481-img balanced cache. Result: DEGENERATE — POPE Overall null, RWQA 0.0, MMBench 0.0; "
            "greedy output gibberish. Root cause: alignment never converged (align loss stayed ~2.38) "
            "— 200 steps on a fresh projector is far too few, so projected vision tokens are noise and "
            "the LM garbles. The loop works; the recipe needs SCALE-UP. Next spec: (1) many more align "
            "steps until align loss actually drops (use the full coco_caption_5k, not the capped 481), "
            "(2) larger balanced cache, (3) consider init=adapt:<small VLM> to skip projector cold-start. "
            "Keep refining this hypothesis — do NOT abandon it."
        ),
    },
]

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ExperimentProposal:
    """Output of the Search Strategist — a proposed next experiment."""
    hypothesis_id:  str
    technique:      str
    model:          str
    rationale:      str
    expected_gain:  str
    gain_axis:      str
    config:         ExperimentConfig | None = None
    # Threshold Monitor state
    consecutive_non_improvements: int = 0
    flag_for_human_review: bool = False
    flag_reason: str = ""
    # Full agent reasoning (tool call trace)
    reasoning_trace: list[dict] = field(default_factory=list)


# ── Tool implementations ──────────────────────────────────────────────────────

def _tool_query_results(model_key: str) -> str:
    """Return MetricsReports for the given model key from the ledger."""
    results = []
    for path in sorted(LEDGER_DIR.glob("*.json")):
        if path.stem.endswith("_preds"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        cfg = data.get("config", {})
        mid = cfg.get("model_id", "")
        if model_key.lower() not in mid.lower() and model_key.lower() not in path.stem.lower():
            continue
        rep = data.get("report", {})
        qs  = {s["metric"]: s["value"] for s in rep.get("quality_scores", [])}
        results.append({
            "experiment_id":      rep.get("experiment_id", "")[:16] + "…",
            "compression":        cfg.get("compression", {}).get("weight_dtype"),
            "input_resolution":   cfg.get("input_resolution"),
            "backend":            cfg.get("runtime_backend"),
            "clip_score":         qs.get("clip_score_mean"),
            "ttft_ms":            rep.get("ttft_ms"),
            "tps":                rep.get("decode_tokens_per_sec"),
            "peak_mem_mb":        rep.get("peak_memory_mb"),
        })
    return json.dumps(results, indent=2)


def _tool_query_frontier() -> str:
    """Return the current Pareto frontier."""
    tracker  = ParetoTracker()
    frontier = tracker.update()
    return json.dumps([
        {
            "experiment_id": p.experiment_id[:16] + "…",
            "label":         p.label,
            "clip_score":    p.clip_score,
            "ttft_ms":       p.ttft_ms,
            "tps":           p.decode_tokens_per_sec,
            "peak_mem_mb":   p.peak_memory_mb,
            "is_baseline":   p.is_baseline,
        }
        for p in frontier
    ], indent=2)


def _tool_propose_experiment(
    hypothesis_id: str = "",
    technique: str = "",
    model: str = "",
    rationale: str = "",
    expected_gain: str = "",
    gain_axis: str = "",
    weight_dtype: str = "",
    runtime_backend: str = "",
    input_resolution: int | None = None,
    n_ctx: int | None = None,
    notes: str | None = None,
    **_extra,          # absorb unknown keys from less-disciplined models
) -> str:
    """
    Validate and enqueue an experiment proposal.
    Returns success confirmation or validation error.
    """
    # Validate required fields — return a clear error rather than crashing.
    # Build an explicit map; do NOT use locals() inside a comprehension. On
    # Python < 3.12 the comprehension has its own scope and locals() won't
    # contain the function arguments, raising KeyError (PEP 709 changed this
    # in 3.12, which is why it only surfaced on the 3.11 CI runner).
    _required = {
        "hypothesis_id":   hypothesis_id,
        "technique":       technique,
        "model":           model,
        "weight_dtype":    weight_dtype,
        "runtime_backend": runtime_backend,
    }
    missing = [name for name, val in _required.items() if not val]
    if missing:
        return json.dumps({
            "status": "error",
            "message": (
                f"Missing required fields: {missing}. "
                "Please call propose_experiment again with all required arguments: "
                "hypothesis_id, technique, model, rationale, expected_gain, gain_axis, "
                "weight_dtype, runtime_backend."
            ),
        })

    try:
        config = ExperimentConfig(
            model_id=_model_id_for_key(model),
            compression=CompressionSpec(weight_dtype=weight_dtype),
            input_resolution=input_resolution,
            n_ctx=n_ctx,
            runtime_backend=runtime_backend,
            decode_strategy="greedy",
            dataset_hash=STAGE_A_HASH,
            target_device_id="iphone16pro-001",
            notes=notes or f"{hypothesis_id}: {technique}",
        )
        proposal_payload = {
            "proposed_at":   datetime.now(timezone.utc).isoformat(),
            "hypothesis_id": hypothesis_id,
            "technique":     technique,
            "model":         model,
            "rationale":     rationale,
            "expected_gain": expected_gain,
            "gain_axis":     gain_axis,
            "experiment_id": config.content_hash(),
            "config":        json.loads(config.model_dump_json()),
        }
        _append_to_queue(proposal_payload)
        return json.dumps({
            "status":        "queued",
            "experiment_id": config.content_hash()[:16] + "…",
            "message": (
                f"Experiment {hypothesis_id} ({technique} on {model}) has been "
                f"validated and added to the experiment queue."
            ),
        })
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


def _model_id_for_key(key: str) -> str:
    MAP = {
        "LFM2-VL-450M":  "LiquidAI/LFM2-VL-450M",
        "SmolVLM-500M":  "HuggingFaceTB/SmolVLM-500M-Instruct",
        "MiniCPM-V-4.6": "openbmb/MiniCPM-V-4.6",
        "FastVLM-0.5B":  "apple/FastVLM-0.5B",
    }
    for k, v in MAP.items():
        if k.lower() in key.lower():
            return v
    return key  # assume already an HF model ID


def _append_to(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    queue = []
    if path.exists():
        try:
            queue = json.loads(path.read_text())
        except Exception:
            queue = []
    queue.append(payload)
    path.write_text(json.dumps(queue, indent=2))


def _append_to_queue(payload: dict) -> None:
    _append_to(QUEUE_FILE, payload)


def _tool_propose_student(
    hypothesis_id: str = "",
    lm: str = "",
    vision: str = "",
    rationale: str = "",
    projector_depth: int = 2,
    projector_hidden: int = 2048,
    init: str = "scratch",
    align_data: str = "coco_caption_5k",
    align_steps: int = 2000,
    distill_teacher: str = "Qwen2.5-VL-3B",
    distill_data: str = "qa_balanced_5k",
    distill_lora_r: int = 16,
    distill_epochs: int = 3,
    eval_n: int = 100,
    notes: str | None = None,
    **_extra,
) -> str:
    """Validate and enqueue a Tier-2 student-CONSTRUCTION proposal (P2-B1, ADR-0012).

    Unlike propose_experiment (a compression config over an existing model), this
    proposes assembling a NEW right-sized student from parts and distilling it. The
    construction loop builds it and writes the result to the ledger.
    """
    _required = {"hypothesis_id": hypothesis_id, "lm": lm, "vision": vision}
    missing = [name for name, val in _required.items() if not val]
    if missing:
        return json.dumps({
            "status": "error",
            "message": (
                f"Missing required fields: {missing}. Call propose_student again with "
                "at least hypothesis_id, lm, vision (HF ids), and a rationale."
            ),
        })
    try:
        spec = StudentSpec(
            lm=lm, vision=vision,
            projector={"type": "mlp", "depth": projector_depth, "hidden": projector_hidden},
            init=init,
            align={"data": align_data, "steps": align_steps},
            distill={"teacher": distill_teacher, "data": distill_data,
                     "lora_r": distill_lora_r, "epochs": distill_epochs},
            eval={"n": eval_n},
            notes=notes or f"{hypothesis_id}: {rationale[:80]}",
        )
        _append_to(CONSTRUCTION_QUEUE, {
            "proposed_at":   datetime.now(timezone.utc).isoformat(),
            "hypothesis_id": hypothesis_id,
            "rationale":     rationale,
            "experiment_id": spec.content_hash(),
            "spec":          json.loads(spec.model_dump_json()),
        })
        return json.dumps({
            "status":        "queued",
            "experiment_id": spec.content_hash()[:16] + "…",
            "message": (
                f"Student-construction proposal {hypothesis_id} "
                f"({lm} + {vision}) validated and queued for the construction loop."
            ),
        })
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


# ── Tool schemas (for Claude API) ────────────────────────────────────────────

TOOLS = [
    {
        "name": "query_results",
        "description": (
            "Return all MetricsReport records from the experiment ledger for a given "
            "model. Use this to inspect what has already been tried and measured."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_key": {
                    "type": "string",
                    "description": "Model identifier, e.g. 'LFM2-VL-450M' or 'SmolVLM-500M'.",
                }
            },
            "required": ["model_key"],
        },
    },
    {
        "name": "query_frontier",
        "description": (
            "Return the current Pareto frontier — the set of non-dominated experiments "
            "across CLIP-score, TTFT, TPS, and peak-memory axes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "propose_experiment",
        "description": (
            "Propose and enqueue the next experiment to run. Validates the config "
            "against the ExperimentConfig schema and writes it to the experiment queue. "
            "Call this ONCE when you have decided on the best next experiment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis_id":   {"type": "string", "description": "e.g. H005 or H006"},
                "technique":       {"type": "string", "description": "Human-readable technique name"},
                "model":           {"type": "string", "description": "e.g. LFM2-VL-450M"},
                "rationale":       {"type": "string", "description": "Why this experiment next, based on results so far"},
                "expected_gain":   {"type": "string", "description": "What improvement is expected and why"},
                "gain_axis":       {"type": "string", "description": "Primary axis: quality | latency | mem | speed+mem | quality+speed"},
                "weight_dtype":    {"type": "string", "description": "fp16 | fp32 | int4 | int8 | q4_0 | q4_k_m | q8_0"},
                "runtime_backend": {"type": "string", "description": "llamacpp_gguf | pytorch_mps"},
                "input_resolution":{"type": ["integer", "null"], "description": "Image side length px, or null for model default"},
                "n_ctx":           {"type": ["integer", "null"], "description": "KV-cache context size (llama.cpp n_ctx). Null = model default (4096). Use 1024 for H005."},
                "notes":           {"type": ["string", "null"], "description": "Optional extra context"},
            },
            "required": [
                "hypothesis_id", "technique", "model", "rationale",
                "expected_gain", "gain_axis", "weight_dtype", "runtime_backend",
            ],
        },
    },
    {
        "name": "propose_student",
        "description": (
            "Propose and enqueue a Tier-2 student-CONSTRUCTION experiment (ADR-0012): "
            "assemble a NEW right-sized VLM from a language-model backbone + a vision "
            "encoder + a projector, then distill it from the teacher. Use this for "
            "construction hypotheses like P2-B1 (the deliverable must derive from the "
            "Qwen2.5-VL-3B lineage, NOT from the LFM2 benchmark). The construction loop "
            "builds the spec and records the result. Call ONCE when decided."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis_id":    {"type": "string", "description": "e.g. P2-B1"},
                "lm":               {"type": "string", "description": "HF id of the LM backbone, e.g. Qwen/Qwen2.5-0.5B-Instruct (same-family as the teacher eases distillation)"},
                "vision":           {"type": "string", "description": "HF id of the vision encoder, e.g. google/siglip-base-patch16-224"},
                "rationale":        {"type": "string", "description": "Why this architecture/recipe next, grounded in the ledger"},
                "projector_depth":  {"type": "integer", "description": "Projector MLP depth (1-4)"},
                "projector_hidden": {"type": "integer", "description": "Projector MLP hidden width"},
                "init":             {"type": "string", "description": "'scratch' or 'adapt:<hf_id>'"},
                "align_data":       {"type": "string", "description": "Alignment data key, e.g. coco_caption_5k"},
                "align_steps":      {"type": "integer", "description": "Projector alignment steps"},
                "distill_teacher":  {"type": "string", "description": "Teacher, e.g. Qwen2.5-VL-3B"},
                "distill_data":     {"type": "string", "description": "Distill data key, e.g. qa_balanced_5k (task-aligned + hard negatives, the P2-D2 fix)"},
                "distill_lora_r":   {"type": "integer", "description": "LoRA rank"},
                "distill_epochs":   {"type": "integer", "description": "Distillation epochs"},
                "eval_n":           {"type": "integer", "description": "Samples per benchmark slice"},
                "notes":            {"type": ["string", "null"], "description": "Optional extra context"},
            },
            "required": ["hypothesis_id", "lm", "vision", "rationale"],
        },
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────

#: Statuses that mean "this hypothesis is no longer a candidate"
_CLOSED_STATUSES = {"CONFIRMED", "NULL_RESULT", "BLOCKED", "REGRESSED", "DEFERRED"}


def _build_system_prompt(hypothesis_table: list[dict] | None = None) -> str:
    table = hypothesis_table or HYPOTHESIS_TABLE

    open_hyps   = [h for h in table if h["status"] not in _CLOSED_STATUSES]
    closed_hyps = [h for h in table if h["status"] in _CLOSED_STATUSES]

    open_json = json.dumps(open_hyps, indent=2)

    closed_lines = "\n".join(
        f"  {h['id']} [{h['status']}]: {h['technique']} on {h['model']}"
        + (f" — {h['result_summary'][:120]}" if h.get("result_summary") else "")
        for h in closed_hyps
    )
    closed_block = (
        "## Closed Hypotheses (already tried or blocked — DO NOT propose these)\n"
        + closed_lines
        if closed_hyps else ""
    )

    baselines_json = json.dumps(PHASE0_BASELINES, indent=2)
    return f"""You are the Search Strategist Agent for a multi-agent VLM optimization system.

Your job is to propose the single best next experiment to run, given:
1. The Phase 0 baselines (what we started with)
2. The hypothesis table (what we planned to try)
3. The experiment ledger (what has already been tried and measured)
4. The current Pareto frontier (what is non-dominated so far)

## Phase 0 Baselines
{baselines_json}

## Open Hypotheses (actionable — NOT_TRIED or IN_PROGRESS; choose from these only)
{open_json}

{closed_block}

## CURRENT PHASE: Phase 2 — produce an edge model FROM Qwen2.5-VL-3B
The project is now in Phase 2. Goal: starting from the open, general-purpose
**Qwen2.5-VL-3B** (NOT an already-optimized model), produce a ~450M-class edge model
that is competitive with the **LFM2-VL-450M benchmark**. LFM2-VL-450M is the
TARGET/YARDSTICK to match — never a student to train.

Same-path baseline to beat (LFM2-VL-450M, fp16 transformers): POPE 86.2, RealWorldQA 42, MMBench 74.
The distillation quality signal is the **MCQ benchmark set** (POPE/RealWorldQA/MMBench),
NOT CLIP-score (P2-1.1: the teacher is not a CLIP leader). Always compare candidates and
the benchmark on the SAME inference path (P2-1.3).

Prioritize Phase 2 hypotheses (phase==2). Phase 1 hypotheses (quantization) are
lower priority now.

## Reasoning Policy (follow this exactly)
1. Query the frontier and recent results first to understand current state.
2. Prefer the highest-expected-gain NOT_TRIED Phase 2 hypothesis that is not invalidated
   by a closed/REGRESSED result.
3. If the last experiment REGRESSED or was not an improvement: do NOT repeat its mistake —
   propose the hypothesis that directly addresses the failure cause.
4. After 3 consecutive non-improvements: flag for human review instead of proposing.
5. Avoid re-proposing any closed hypothesis (CONFIRMED/NULL_RESULT/BLOCKED/REGRESSED).

## Tier-2 (code-requiring) hypotheses — Phase 2
All Phase 2 hypotheses are Tier-2 (new training/architecture code, HLD §6.3). There
are now TWO kinds, with different tools:

- **CONSTRUCTION hypotheses (e.g. P2-B1) → call `propose_student`.** Per ADR-0012 the
  system now BUILDS students: you emit a StudentSpec (lm + vision + projector + align +
  distill + eval) and the construction loop assembles and distills it automatically.
  The deliverable must derive from the Qwen2.5-VL-3B lineage (e.g. lm=Qwen/Qwen2.5-0.5B-
  Instruct, vision=google/siglip-base-patch16-224), NOT from the LFM2 benchmark. Use
  distill_data=qa_balanced_5k (task-aligned + balanced hard negatives — the P2-D2 fix).
- **Other Tier-2 hypotheses → call `propose_experiment`** with weight_dtype/runtime_backend
  set to the intended DEPLOY target and the technique + rationale in their fields.

Recommend the single best next hypothesis with a concrete rationale grounded in the ledger.

## Quality Gates (any proposal must expect to pass these)
- POPE accuracy ≥ 89.0% (must not regress below the LFM2 benchmark 86.2 — see P2-D1)
- Peak memory ≤ 3000 MB on Mac proxy; edge target ≤ LFM2 footprint
- No OOM crash

## Key constraints
- LFM2-VL-450M is the BENCHMARK, not a student. Do NOT propose distilling/fine-tuning
  INTO LFM2 or other already-edge-optimized models (P2-D1 regressed and it violates the
  Phase 2 premise). The edge model's lineage must be Qwen2.5-VL-3B.
- Caption-only distillation REGRESSED (P2-D1): it teaches captioning, not the measured
  MCQ skill, and causes forgetting. Distill the skill we MEASURE (grounded VQA/MCQ) + rehearsal.
- Architecture budget (measured): Qwen2.5-VL-3B vision encoder 669M + embeddings 311M =
  980M alone — already > 2× the 450M target. Pruning LM layers cannot hit the budget without
  also replacing the vision encoder and shrinking the vocab.

## Output
Use the available tools to query current state, reason through the options, and
call propose_experiment exactly once with your best proposal. Be concrete and cite
evidence from the ledger in your rationale.
"""


# ── LLM backend abstraction ───────────────────────────────────────────────────
# Normalised turn: (stop_reason, text, tool_calls)
# stop_reason: "done" | "tool_use"
# tool_calls:  [{"id": str, "name": str, "input": dict}]

def _openai_tools_from_anthropic(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool schema → OpenAI function-calling schema."""
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"],
            },
        })
    return out


class _AnthropicBackend:
    def __init__(self, model: str, api_key: str):
        if not _HAS_ANTHROPIC:
            raise ImportError("pip install anthropic")
        self.client = _anthropic_mod.Anthropic(api_key=api_key)
        self.model  = model

    def chat(self, messages: list[dict], system: str, tools: list[dict]):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )
        text_parts = [b.text for b in response.content if b.type == "text"]
        tool_calls = [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in response.content if b.type == "tool_use"
        ]
        stop = "tool_use" if response.stop_reason == "tool_use" else "done"
        return stop, "\n".join(text_parts), tool_calls

    def append_assistant(self, messages: list[dict], text: str, tool_calls: list[dict],
                         _raw_response=None):
        """Append the assistant turn in Anthropic message format."""
        content = []
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]})
        messages.append({"role": "assistant", "content": content})

    def append_tool_results(self, messages: list[dict], results: list[dict]):
        """Append tool results in Anthropic format."""
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r["id"], "content": r["result"]}
                for r in results
            ],
        })


def _react_system_suffix(tools: list[dict]) -> str:
    """
    Describe tools in a ReAct-style JSON format for models that don't support
    the native tool-calling API. Appended to the system prompt.
    """
    tool_descs = []
    for t in tools:
        props = t["input_schema"].get("properties", {})
        req   = t["input_schema"].get("required", [])
        params = ", ".join(
            f"{k} ({'required' if k in req else 'optional'})"
            for k in props
        )
        tool_descs.append(f"- {t['name']}({params}): {t['description']}")

    return """

## Tool use (JSON mode)
This model does not use the native function-calling API.
Instead, output tool calls as a JSON block in your reply:

```json
{
  "thought": "brief reason for calling this tool",
  "action": "<tool_name>",
  "action_input": { ... }
}
```

After receiving the tool result, you may call another tool or output your final
answer as plain text (no JSON block). Call propose_experiment last, once.

Available tools:
""" + "\n".join(tool_descs)


class _OpenAICompatibleBackend:
    """
    Works with any OpenAI-compatible endpoint: llama.cpp server, Ollama,
    LM Studio, vLLM, etc.

    For llama.cpp + Qwen2.5 (default local backend):
        base_url = "http://localhost:8080/v1"
        model    = "qwen2.5-7b-instruct"
        Launch llama-server with --jinja so the OpenAI `tools` API works
        natively (Qwen2.5 is tool-trained — no ReAct fallback needed).

    For Ollama:
        base_url = "http://localhost:11434/v1"
        model    = "gemma3"  (tool-less → uses ReAct JSON fallback)

    Auto-detects whether the model supports native tool calling.
    Falls back to a ReAct JSON-mode approach for models/servers that
    return HTTP 400 on the `tools` parameter (e.g. gemma3 via Ollama).
    """

    def __init__(self, model: str, base_url: str, api_key: str = "local",
                 verbose: bool = True):
        if not _HAS_OPENAI:
            raise ImportError("pip install openai")
        self.client      = _openai_mod.OpenAI(base_url=base_url, api_key=api_key)
        self.model       = model
        self.verbose     = verbose
        self._oai_tools  = _openai_tools_from_anthropic(TOOLS)
        self._react_mode = False   # set True after first "does not support tools" 400

    def chat(self, messages: list[dict], system: str, tools: list[dict]):
        # Inject system prompt (+ ReAct suffix if needed) as first message
        sys_content = system + (_react_system_suffix(tools) if self._react_mode else "")
        msgs = messages
        if not msgs or msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": sys_content}] + messages
        else:
            # replace system message with updated content
            msgs = [{"role": "system", "content": sys_content}] + messages[1:]

        if self._react_mode:
            return self._chat_react(msgs)
        else:
            return self._chat_tools(msgs, tools)

    def _chat_tools(self, msgs: list[dict], tools: list[dict]):
        """Try native tool calling; fall back to ReAct on 400."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                tools=self._oai_tools,
                tool_choice="auto",
            )
        except Exception as e:
            err_str = str(e)
            if "does not support tools" in err_str or "400" in err_str:
                print(f"    [backend] model doesn't support tools API — switching to ReAct JSON mode")
                self._react_mode = True
                # Rebuild msgs with ReAct suffix now injected into the system message
                sys_msg = msgs[0]["content"] + _react_system_suffix(TOOLS)
                msgs_react = [{"role": "system", "content": sys_msg}] + msgs[1:]
                return self._chat_react(msgs_react)
            raise

        choice  = response.choices[0]
        message = choice.message
        text    = message.content or ""
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    inp = json.loads(tc.function.arguments)
                except Exception:
                    inp = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "input": inp})
        stop = "tool_use" if tool_calls else "done"
        return stop, text, tool_calls

    def _chat_react(self, msgs: list[dict]):
        """
        ReAct JSON-mode: parse tool calls from markdown ```json blocks.
        Works with any model that can follow JSON instructions.
        """
        import re, uuid
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
        )
        text = response.choices[0].message.content or ""
        if self.verbose:
            print(f"    [backend raw] {text[:300].replace(chr(10), ' ')}")

        # Extract ```json ... ``` block
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not m:
            # Try raw JSON object anywhere in the text
            m = re.search(r"\{[^{}]*\"action\"\s*:[^{}]*\}", text, re.DOTALL)

        if m:
            try:
                obj = json.loads(m.group(1) if "```" in (m.group(0) or "") else m.group(0))
                action = obj.get("action", "")
                if action in {t["name"] for t in TOOLS}:
                    tc = {
                        "id":    f"react_{uuid.uuid4().hex[:8]}",
                        "name":  action,
                        "input": obj.get("action_input", {}),
                    }
                    # strip the json block from visible text
                    visible = text[:m.start()].strip()
                    return "tool_use", visible, [tc]
            except json.JSONDecodeError:
                pass

        # No tool call found — treat as final answer
        return "done", text, []

    def append_assistant(self, messages: list[dict], text: str, tool_calls: list[dict],
                         _raw_response=None):
        """Append the assistant turn in OpenAI / ReAct format."""
        if self._react_mode:
            # In ReAct mode, the assistant message is just the text (no tool_calls field)
            content = text or ""
            if tool_calls:
                # Re-embed the JSON block so the history is coherent
                tc = tool_calls[0]
                content += f'\n```json\n{json.dumps({"action": tc["name"], "action_input": tc["input"]}, indent=2)}\n```'
            messages.append({"role": "assistant", "content": content})
        else:
            tc_list = None
            if tool_calls:
                tc_list = [
                    {
                        "id":       tc["id"],
                        "type":     "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])},
                    }
                    for tc in tool_calls
                ]
            messages.append({"role": "assistant", "content": text or None, "tool_calls": tc_list})

    def append_tool_results(self, messages: list[dict], results: list[dict]):
        """Append tool results — plain user message in ReAct mode, tool messages otherwise."""
        if self._react_mode:
            parts = []
            for r in results:
                parts.append(f"Tool result for `{r.get('name', 'tool')}`:\n{r['result']}")
            messages.append({"role": "user", "content": "\n\n".join(parts)})
        else:
            for r in results:
                messages.append({"role": "tool", "tool_call_id": r["id"], "content": r["result"]})


# ── Agent ─────────────────────────────────────────────────────────────────────

class SearchStrategist:
    """
    Proposes the next VLM optimization experiment using an LLM with tool use.

    Supports three backends — pick one:

    1. llama.cpp server + Qwen2.5 (local default; native tool-calling):
       agent = SearchStrategist(backend="llamacpp")        # default: qwen2.5-7b-instruct
       # Requires llama-server running with --jinja. See scripts/start_strategist_llm.sh.

    2. Anthropic API (used when ANTHROPIC_API_KEY is set):
       agent = SearchStrategist(backend="anthropic")

    3. Ollama or any other OpenAI-compatible server:
       agent = SearchStrategist(backend="ollama")          # default: gemma3 (ReAct fallback)
       agent = SearchStrategist(backend="openai_compat",
                                model="...",
                                base_url="http://localhost:1234/v1")

    Args:
        backend:    "auto" | "local"/"llamacpp" | "api"/"anthropic" | "ollama" | "openai_compat"
                    DEFAULT IS LOCAL. "auto" resolves to llamacpp unless
                    STRATEGIST_BACKEND is set (api/anthropic/ollama/...). API is
                    opt-in — ANTHROPIC_API_KEY alone does NOT switch to it (ADR-0013).
        model:      Model name. Defaults: Anthropic → "claude-sonnet-4-5",
                    llamacpp → "qwen2.5-7b-instruct", Ollama → "gemma3".
        api_key:    Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        base_url:   Base URL for OpenAI-compatible server.
                    Default for "llamacpp": "http://localhost:8080/v1".
        max_rounds: Max tool-call rounds before giving up (default 6).
        verbose:    Print reasoning trace to stdout (default True).
    """

    def __init__(
        self,
        backend:    str = "auto",
        model:      str | None = None,
        api_key:    str | None = None,
        base_url:   str | None = None,
        max_rounds: int = 6,
        verbose:    bool = True,
    ):
        self.max_rounds = max_rounds
        self.verbose    = verbose
        self._last_proposal: ExperimentProposal | None = None

        # Resolve backend — default LOCAL; API is opt-in via backend= or
        # STRATEGIST_BACKEND env (ADR-0013). Presence of ANTHROPIC_API_KEY alone
        # does NOT switch to the API.
        backend = _resolve_backend_name(backend)

        if backend == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set it or use backend='llamacpp' for a local model."
                )
            self._backend = _AnthropicBackend(
                model=model or "claude-sonnet-4-5",
                api_key=key,
            )

        elif backend in ("llamacpp", "ollama", "openai_compat"):
            if backend == "ollama":
                url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
                mdl = model or os.environ.get("OLLAMA_MODEL", "gemma3")
            else:   # llamacpp (default) and generic openai_compat
                url = base_url or os.environ.get("LLAMACPP_BASE_URL", LLAMACPP_BASE_URL)
                mdl = model or os.environ.get("LLAMACPP_MODEL", LLAMACPP_MODEL)
            self._backend = _OpenAICompatibleBackend(
                model=mdl,
                base_url=url,
                api_key=api_key or "local",
                verbose=self.verbose,
            )
            if self.verbose:
                print(f"  [SearchStrategist] backend={backend}  model={mdl}  url={url}")

        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'anthropic', 'ollama', or 'openai_compat'.")

    # ── Public API ────────────────────────────────────────────────────────────

    def propose_next(self) -> ExperimentProposal:
        """
        Run the agent loop and return an ExperimentProposal.
        Raises RuntimeError if the agent fails to propose within max_rounds.
        """
        consecutive_non_improvements = self._count_consecutive_non_improvements()

        if consecutive_non_improvements >= 3:
            proposal = ExperimentProposal(
                hypothesis_id="HUMAN_REVIEW",
                technique="N/A",
                model="N/A",
                rationale=(
                    f"{consecutive_non_improvements} consecutive experiments without "
                    "Pareto improvement. Flagging for human review per policy."
                ),
                expected_gain="N/A",
                gain_axis="N/A",
                consecutive_non_improvements=consecutive_non_improvements,
                flag_for_human_review=True,
                flag_reason=(
                    f"Policy: flag after 3 consecutive non-improvements. "
                    f"Count is now {consecutive_non_improvements}."
                ),
            )
            self._last_proposal = proposal
            return proposal

        system   = _build_system_prompt()
        messages: list[dict] = [
            {
                "role":    "user",
                "content": (
                    "Please analyse the current experiment state and propose the "
                    "single best next experiment to run. Use the available tools to "
                    "query the frontier and ledger before deciding."
                ),
            }
        ]

        reasoning_trace: list[dict] = []
        proposed: ExperimentProposal | None = None

        for round_idx in range(self.max_rounds):
            if self.verbose:
                print(f"\n  [SearchStrategist] Round {round_idx + 1}/{self.max_rounds}")

            stop, text, tool_calls = self._backend.chat(messages, system, TOOLS)

            reasoning_trace.append({
                "round":       round_idx + 1,
                "stop_reason": stop,
                "text":        text[:200] if text else "",
                "tool_calls":  [{"name": tc["name"], "input": tc["input"]} for tc in tool_calls],
            })

            self._backend.append_assistant(messages, text, tool_calls)

            if stop == "done":
                break

            # Process tool calls
            results = []
            for tc in tool_calls:
                tool_name  = tc["name"]
                tool_input = tc["input"]

                if self.verbose:
                    print(f"    → tool: {tool_name}({json.dumps(tool_input)[:80]}…)")

                if tool_name == "query_results":
                    result = _tool_query_results(tool_input["model_key"])
                elif tool_name == "query_frontier":
                    result = _tool_query_frontier()
                elif tool_name == "propose_experiment":
                    result = _tool_propose_experiment(**tool_input)
                    try:
                        result_data = json.loads(result)
                        if result_data.get("status") == "queued":
                            proposed = ExperimentProposal(
                                hypothesis_id=tool_input.get("hypothesis_id", "?"),
                                technique=tool_input.get("technique", "?"),
                                model=tool_input.get("model", "?"),
                                rationale=tool_input.get("rationale", ""),
                                expected_gain=tool_input.get("expected_gain", ""),
                                gain_axis=tool_input.get("gain_axis", ""),
                                config=None,
                                consecutive_non_improvements=consecutive_non_improvements,
                                reasoning_trace=reasoning_trace,
                            )
                    except Exception:
                        pass
                elif tool_name == "propose_student":
                    result = _tool_propose_student(**tool_input)
                    try:
                        if json.loads(result).get("status") == "queued":
                            proposed = ExperimentProposal(
                                hypothesis_id=tool_input.get("hypothesis_id", "?"),
                                technique="student-construction (ADR-0012)",
                                model=f"{tool_input.get('lm','?')} + {tool_input.get('vision','?')}",
                                rationale=tool_input.get("rationale", ""),
                                expected_gain="edge-size student derived from the 3B lineage",
                                gain_axis="size+quality",
                                config=None,
                                consecutive_non_improvements=consecutive_non_improvements,
                                reasoning_trace=reasoning_trace,
                            )
                    except Exception:
                        pass
                else:
                    result = json.dumps({"error": f"Unknown tool: {tool_name}"})

                results.append({"id": tc["id"], "name": tool_name, "result": result})

            self._backend.append_tool_results(messages, results)

        if proposed is None:
            raise RuntimeError(
                f"SearchStrategist failed to call propose_experiment within "
                f"{self.max_rounds} rounds. Check verbose output."
            )

        proposed.reasoning_trace = reasoning_trace
        self._last_proposal = proposed
        return proposed

    def print_report(self) -> None:
        """Print the last proposal and its reasoning summary."""
        p = self._last_proposal
        if p is None:
            print("  (no proposal yet — call propose_next() first)")
            return

        print("\n" + "═" * 72)
        print("  Search Strategist Proposal")
        print("═" * 72)
        if p.flag_for_human_review:
            print(f"  ⚠️  HUMAN REVIEW FLAGGED")
            print(f"  Reason: {p.flag_reason}")
        else:
            print(f"  Hypothesis:    {p.hypothesis_id}  —  {p.technique}")
            print(f"  Model:         {p.model}")
            print(f"  Expected gain: {p.expected_gain}  [{p.gain_axis}]")
            print(f"  Rationale:")
            for line in p.rationale.split(". "):
                sentence = line.strip().rstrip(".")
                if sentence:
                    print(f"    • {sentence}.")
        print(f"  Consec. non-improvements: {p.consecutive_non_improvements}")
        print("═" * 72)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _count_consecutive_non_improvements(self) -> int:
        """
        Count how many of the most recent experiments (in ledger order) failed to
        advance the Pareto frontier.
        """
        tracker = ParetoTracker()
        tracker.update()
        dominated_ids = {
            p.experiment_id for p in tracker.points
            if not p.is_pareto_frontier and not p.is_baseline
        }
        frontier_ids = {
            p.experiment_id for p in tracker.points
            if p.is_pareto_frontier and not p.is_baseline
        }

        ledger_files = sorted(
            (p for p in LEDGER_DIR.glob("*.json") if not p.stem.endswith("_preds")),
            key=lambda p: p.stat().st_mtime,
        )

        consecutive = 0
        for path in reversed(ledger_files):
            try:
                data   = json.loads(path.read_text())
                exp_id = data["report"]["experiment_id"]
            except Exception:
                continue
            if exp_id in frontier_ids:
                break
            if exp_id in dominated_ids:
                consecutive += 1
        return consecutive


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Usage:
    #   python3 agents/search_strategist.py                          # auto (llamacpp local, or anthropic if key set)
    #   python3 agents/search_strategist.py llamacpp                 # llama.cpp server (default: qwen2.5-7b-instruct)
    #   python3 agents/search_strategist.py llamacpp qwen2.5-32b     # llama.cpp with a specific model
    #   python3 agents/search_strategist.py anthropic               # force Anthropic API
    #   python3 agents/search_strategist.py ollama                  # Ollama (default: gemma3, ReAct fallback)
    #   LLAMACPP_BASE_URL=http://localhost:8080/v1 python3 agents/search_strategist.py llamacpp
    #
    # Start the local LLM first:  ./scripts/start_strategist_llm.sh

    backend = sys.argv[1] if len(sys.argv) > 1 else "auto"
    model   = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n  backend={backend}  model={model or '(default)'}")
    agent    = SearchStrategist(backend=backend, model=model, verbose=True)
    proposal = agent.propose_next()
    agent.print_report()

    # Dump queue entry
    if QUEUE_FILE.exists():
        queue = json.loads(QUEUE_FILE.read_text())
        if queue:
            print(f"\n  Latest queue entry:")
            latest = queue[-1]
            print(json.dumps(latest, indent=4))
