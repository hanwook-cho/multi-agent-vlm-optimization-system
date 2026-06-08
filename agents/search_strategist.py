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
from services.pareto_tracker import ParetoTracker, PHASE0_BASELINES, DEFAULT_AXES

LEDGER_DIR  = PROJECT_ROOT / "artifacts" / "experiment_ledger"
QUEUE_FILE  = PROJECT_ROOT / "artifacts" / "experiment_queue.json"

STAGE_A_HASH = "e2128ae022b3720375d7c866a037b6d8ec4b399ff92cb59e6065ec9fb7f3e29f"

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
        "status": "NOT_TRIED",
        "result_summary": "",
    },
    {
        "id": "H006",
        "technique": "GPTQ INT4 LM backbone",
        "model": "LFM2-VL-450M",
        "expected_gain": "CLIP +0–1, POPE neutral (quality-preserving INT4)",
        "gain_axis": "quality",
        "status": "NOT_TRIED",
        "result_summary": "",
    },
    {
        "id": "H007",
        "technique": "FastVLM INT4 MLX build",
        "model": "FastVLM-0.5B",
        "expected_gain": "TTFT -90%, Mem -65% (vs fp16 baseline)",
        "gain_axis": "latency+mem",
        "status": "NOT_TRIED",
        "result_summary": "",
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
    # Validate required fields — return a clear error rather than crashing
    missing = [f for f in ("hypothesis_id","technique","model","weight_dtype","runtime_backend")
               if not locals()[f]]
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


def _append_to_queue(payload: dict) -> None:
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    queue = []
    if QUEUE_FILE.exists():
        try:
            queue = json.loads(QUEUE_FILE.read_text())
        except Exception:
            queue = []
    queue.append(payload)
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))


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
]

# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    hyp_json = json.dumps(HYPOTHESIS_TABLE, indent=2)
    baselines_json = json.dumps(PHASE0_BASELINES, indent=2)
    return f"""You are the Search Strategist Agent for a multi-agent VLM optimization system.

Your job is to propose the single best next experiment to run, given:
1. The Phase 0 baselines (what we started with)
2. The hypothesis table (what we planned to try)
3. The experiment ledger (what has already been tried and measured)
4. The current Pareto frontier (what is non-dominated so far)

## Phase 0 Baselines
{baselines_json}

## Hypothesis Table
{hyp_json}

## Reasoning Policy (follow this exactly)
1. Query the frontier and recent results first to understand current state.
2. Identify the highest-expected-gain hypothesis that has NOT been tried (status NOT_TRIED).
3. If the last experiment was a Pareto improvement: consider a variation of that technique.
4. If the last experiment was NOT an improvement: move to the next untried technique.
5. After 3 consecutive non-improvements: flag for human review instead of proposing.
6. Avoid re-proposing any experiment whose experiment_id is already in the ledger.

## Quality Gates (any proposal must expect to pass these)
- POPE accuracy ≥ 89.0%
- CLIP-score ≥ 25.0
- Peak memory ≤ 3000 MB
- No OOM crash on 5 test images

## Key constraints
- Phase 1 only uses post-training techniques (quantization, input changes, ctx-size).
- No training runs, no distillation — those are Phase 2.
- iPhone runtime is llamacpp_gguf. Mac quality proxy uses pytorch_mps.
- IMPORTANT: Input resize (input_resolution != null) does NOT reduce TTFT on the
  llamacpp_gguf path — H003 proved this. Do not propose input resize as a latency
  technique for GGUF models.
- IMPORTANT: Q4_0 mmproj quantization is BLOCKED (all tooling paths fail). Do not
  propose H004 (Q4_0 mmproj) — it cannot be executed.

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
    Works with any OpenAI-compatible endpoint: Ollama, LM Studio, vLLM, etc.

    For Ollama (default):
        base_url = "http://localhost:11434/v1"
        api_key  = "ollama"   (ignored by Ollama but required by the SDK)
        model    = "gemma3"  (or "llama3.2", "qwen2.5", ...)

    Auto-detects whether the model supports native tool calling.
    Falls back to a ReAct JSON-mode approach for models like gemma3 that
    don't expose tools via the OpenAI endpoint.
    """

    def __init__(self, model: str, base_url: str, api_key: str = "ollama"):
        if not _HAS_OPENAI:
            raise ImportError("pip install openai")
        self.client      = _openai_mod.OpenAI(base_url=base_url, api_key=api_key)
        self.model       = model
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
        # Show first 300 chars so we can see if the model is on track
        print(f"    [gemma3 raw] {text[:300].replace(chr(10),' ')}")

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

    Supports two backends — pick one:

    1. Anthropic API (default if ANTHROPIC_API_KEY is set):
       agent = SearchStrategist()                          # auto-detect
       agent = SearchStrategist(backend="anthropic")

    2. Local model via Ollama (or any OpenAI-compatible server):
       agent = SearchStrategist(backend="ollama")          # default: gemma3
       agent = SearchStrategist(backend="ollama",
                                model="llama3.2:3b")
       agent = SearchStrategist(backend="openai_compat",
                                model="...",
                                base_url="http://localhost:1234/v1")

    Args:
        backend:    "auto" | "anthropic" | "ollama" | "openai_compat"
                    "auto" tries Anthropic first (if ANTHROPIC_API_KEY set),
                    then falls back to Ollama.
        model:      Model name. Defaults: Anthropic → "claude-sonnet-4-5",
                    Ollama → "gemma3".
        api_key:    Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        base_url:   Base URL for OpenAI-compatible server.
                    Default for "ollama": "http://localhost:11434/v1".
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

        # Resolve backend
        if backend == "auto":
            if api_key or os.environ.get("ANTHROPIC_API_KEY"):
                backend = "anthropic"
            else:
                backend = "ollama"

        if backend == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set it or use backend='ollama' for a local model."
                )
            self._backend = _AnthropicBackend(
                model=model or "claude-sonnet-4-5",
                api_key=key,
            )

        elif backend in ("ollama", "openai_compat"):
            url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            mdl = model or os.environ.get("OLLAMA_MODEL", "gemma3")
            self._backend = _OpenAICompatibleBackend(
                model=mdl,
                base_url=url,
                api_key=api_key or "ollama",
            )
            if self.verbose:
                print(f"  [SearchStrategist] backend=ollama  model={mdl}  url={url}")

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
                if line.strip():
                    print(f"    • {line.strip()}.")
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
    #   python3 agents/search_strategist.py                    # auto-detect backend
    #   python3 agents/search_strategist.py anthropic          # force Anthropic API
    #   python3 agents/search_strategist.py ollama             # Ollama (default: gemma3)
    #   python3 agents/search_strategist.py ollama llama3.2    # Ollama with specific model
    #   OLLAMA_MODEL=qwen2.5 python3 agents/search_strategist.py ollama

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
