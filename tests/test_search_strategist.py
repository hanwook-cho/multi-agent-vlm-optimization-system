"""
Unit tests for agents/search_strategist.py

Covers:
  1. _openai_tools_from_anthropic() — schema conversion
  2. ReAct JSON parser — _chat_react() regex extraction
  3. _build_system_prompt() — closed-hypothesis filtering
  4. _tool_propose_experiment() — missing-field validation

These tests are fully offline — no LLM calls, no file I/O.
"""

from __future__ import annotations

import json
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest

# ── Import targets ────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.search_strategist import (
    TOOLS,
    _CLOSED_STATUSES,
    _build_system_prompt,
    _openai_tools_from_anthropic,
    _tool_propose_experiment,
    HYPOTHESIS_TABLE,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. _openai_tools_from_anthropic()
# ═════════════════════════════════════════════════════════════════════════════

class TestOpenAIToolsFromAnthropic:

    def test_output_length_matches_input(self):
        result = _openai_tools_from_anthropic(TOOLS)
        assert len(result) == len(TOOLS)

    def test_top_level_type_is_function(self):
        for item in _openai_tools_from_anthropic(TOOLS):
            assert item["type"] == "function"

    def test_function_wrapper_contains_name_description_parameters(self):
        for item in _openai_tools_from_anthropic(TOOLS):
            fn = item["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_name_and_description_preserved(self):
        result = _openai_tools_from_anthropic(TOOLS)
        for orig, converted in zip(TOOLS, result):
            assert converted["function"]["name"] == orig["name"]
            assert converted["function"]["description"] == orig["description"]

    def test_parameters_is_input_schema(self):
        result = _openai_tools_from_anthropic(TOOLS)
        for orig, converted in zip(TOOLS, result):
            assert converted["function"]["parameters"] == orig["input_schema"]

    def test_empty_input_returns_empty_list(self):
        assert _openai_tools_from_anthropic([]) == []

    def test_single_tool_roundtrip(self):
        single = [
            {
                "name": "my_tool",
                "description": "Does something",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            }
        ]
        result = _openai_tools_from_anthropic(single)
        assert result[0]["function"]["parameters"]["required"] == ["x"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. ReAct JSON parser
#    We test the regex extraction logic directly rather than mocking the full
#    OpenAI client, since _chat_react() is tightly coupled to the HTTP call.
#    Extract the parsing logic into a helper and test that.
# ═════════════════════════════════════════════════════════════════════════════

def _parse_react_block(text: str) -> tuple[str, dict] | None:
    """
    Mirror of the extraction logic inside _OpenAICompatibleBackend._chat_react().
    Returns (action, action_input) or None if nothing found.
    """
    tool_names = {t["name"] for t in TOOLS}

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"\{[^{}]*\"action\"\s*:[^{}]*\}", text, re.DOTALL)

    if m:
        raw = m.group(1) if "```" in (m.group(0) or "") else m.group(0)
        try:
            obj = json.loads(raw)
            action = obj.get("action", "")
            if action in tool_names:
                return action, obj.get("action_input", {})
        except json.JSONDecodeError:
            pass

    return None


class TestReActParser:

    def test_fenced_json_block_parsed(self):
        text = '''
I will query the results first.

```json
{
  "thought": "Let me check the ledger",
  "action": "query_results",
  "action_input": {"model_key": "LFM2-VL-450M"}
}
```
'''
        result = _parse_react_block(text)
        assert result is not None
        action, inputs = result
        assert action == "query_results"
        assert inputs == {"model_key": "LFM2-VL-450M"}

    def test_bare_json_object_fallback(self):
        # The bare-JSON fallback regex uses [^{}]* so it cannot handle nested
        # braces in action_input.  query_frontier takes no args — use null here.
        text = '{"thought": "ok", "action": "query_frontier", "action_input": null}'
        result = _parse_react_block(text)
        assert result is not None
        assert result[0] == "query_frontier"
        # action_input is None from JSON null; the caller handles missing key
        assert result[1] is None or result[1] == {}

    def test_bare_json_fallback_cannot_handle_nested_braces(self):
        # Known limitation: bare JSON with a nested object in action_input is
        # not parsed by the fallback regex.  Models must use the fenced block.
        text = '{"action": "query_results", "action_input": {"model_key": "LFM2"}}'
        # This will return None because [^{}]* rejects the inner {}
        result = _parse_react_block(text)
        assert result is None  # expected — use fenced block for nested args

    def test_unknown_action_returns_none(self):
        text = '```json\n{"action": "do_something_unknown", "action_input": {}}\n```'
        result = _parse_react_block(text)
        assert result is None

    def test_invalid_json_returns_none(self):
        text = "```json\n{broken json here}\n```"
        result = _parse_react_block(text)
        assert result is None

    def test_no_json_block_returns_none(self):
        text = "I have analysed the situation and recommend H005."
        result = _parse_react_block(text)
        assert result is None

    def test_propose_experiment_action_parsed(self):
        text = '''
```json
{
  "thought": "H005 is the best next step",
  "action": "propose_experiment",
  "action_input": {
    "hypothesis_id": "H005",
    "technique": "ctx-size reduction",
    "model": "LFM2-VL-450M",
    "rationale": "Reduce KV-cache allocation",
    "expected_gain": "Mem -15%",
    "gain_axis": "mem",
    "weight_dtype": "int4",
    "runtime_backend": "llamacpp_gguf"
  }
}
```
'''
        result = _parse_react_block(text)
        assert result is not None
        action, inputs = result
        assert action == "propose_experiment"
        assert inputs["hypothesis_id"] == "H005"
        assert inputs["weight_dtype"] == "int4"

    def test_multiline_thought_field_handled(self):
        text = '''
```json
{
  "thought": "First I need to understand\\nwhat has been tried already",
  "action": "query_results",
  "action_input": {"model_key": "SmolVLM-500M"}
}
```
'''
        result = _parse_react_block(text)
        assert result is not None
        assert result[0] == "query_results"


# ═════════════════════════════════════════════════════════════════════════════
# 3. _build_system_prompt() — closed-hypothesis filtering
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildSystemPrompt:

    def test_closed_hypotheses_not_in_open_section(self):
        prompt = _build_system_prompt()
        # The open JSON section must not contain any CONFIRMED/NULL_RESULT/BLOCKED entry
        # Find the open JSON block (between the "Open Hypotheses" header and the next ##)
        open_section_match = re.search(
            r"## Open Hypotheses.*?\n(\[.*?\])\n",
            prompt,
            re.DOTALL,
        )
        assert open_section_match, "Open Hypotheses JSON block not found in prompt"
        open_json = json.loads(open_section_match.group(1))
        statuses_in_open = {h["status"] for h in open_json}
        assert statuses_in_open.isdisjoint(_CLOSED_STATUSES), (
            f"Closed statuses {statuses_in_open & _CLOSED_STATUSES} found in open section"
        )

    def test_only_not_tried_in_open_section(self):
        prompt = _build_system_prompt()
        open_section_match = re.search(
            r"## Open Hypotheses.*?\n(\[.*?\])\n",
            prompt,
            re.DOTALL,
        )
        assert open_section_match
        open_json = json.loads(open_section_match.group(1))
        # Open = actionable = NOT closed. NOT_TRIED and IN_PROGRESS (a hypothesis
        # tried once but still being refined, e.g. P2-B1) are both valid here.
        closed = {"CONFIRMED", "NULL_RESULT", "BLOCKED", "REGRESSED", "DEFERRED"}
        for h in open_json:
            assert h["status"] not in closed

    def test_closed_hypotheses_appear_in_closed_block(self):
        prompt = _build_system_prompt()
        # H001 is CONFIRMED — must appear somewhere in the prompt (as context)
        assert "H001" in prompt
        assert "CONFIRMED" in prompt or "closed" in prompt.lower()

    def test_custom_table_respects_filter(self):
        custom = [
            {"id": "X001", "status": "NOT_TRIED",  "technique": "A", "model": "M", "result_summary": ""},
            {"id": "X002", "status": "CONFIRMED",   "technique": "B", "model": "M", "result_summary": "done"},
            {"id": "X003", "status": "NULL_RESULT", "technique": "C", "model": "M", "result_summary": "null"},
        ]
        prompt = _build_system_prompt(hypothesis_table=custom)
        open_match = re.search(r"## Open Hypotheses.*?\n(\[.*?\])\n", prompt, re.DOTALL)
        assert open_match
        open_json = json.loads(open_match.group(1))
        ids = [h["id"] for h in open_json]
        assert ids == ["X001"]

    def test_all_closed_table_produces_empty_open_section(self):
        all_closed = [
            {"id": "H001", "status": "CONFIRMED",   "technique": "A", "model": "M", "result_summary": ""},
            {"id": "H003", "status": "NULL_RESULT",  "technique": "B", "model": "M", "result_summary": ""},
            {"id": "H004", "status": "BLOCKED",      "technique": "C", "model": "M", "result_summary": ""},
        ]
        prompt = _build_system_prompt(hypothesis_table=all_closed)
        open_match = re.search(r"## Open Hypotheses.*?\n(\[.*?\])\n", prompt, re.DOTALL)
        assert open_match
        open_json = json.loads(open_match.group(1))
        assert open_json == []


# ═════════════════════════════════════════════════════════════════════════════
# 4. _tool_propose_experiment() — validation
# ═════════════════════════════════════════════════════════════════════════════

class TestToolProposeExperiment:

    def test_missing_required_fields_returns_error(self):
        result = json.loads(_tool_propose_experiment())
        assert result["status"] == "error"
        assert "Missing required fields" in result["message"]

    def test_missing_single_field_names_it(self):
        result = json.loads(_tool_propose_experiment(
            hypothesis_id="H005",
            technique="ctx-size reduction",
            model="LFM2-VL-450M",
            weight_dtype="int4",
            # runtime_backend missing
        ))
        assert result["status"] == "error"
        assert "runtime_backend" in result["message"]

    def test_extra_kwargs_do_not_crash(self):
        # Models sometimes pass unexpected fields — must absorb them
        result = json.loads(_tool_propose_experiment(
            hypothesis_id="H005",
            technique="ctx-size reduction",
            model="LFM2-VL-450M",
            weight_dtype="int4",
            runtime_backend="llamacpp_gguf",
            unknown_field_from_model="whatever",
            another_extra=42,
        ))
        # Should not raise — result is either queued or a schema validation error
        assert result["status"] in ("queued", "error")

    def test_invalid_weight_dtype_returns_error(self):
        result = json.loads(_tool_propose_experiment(
            hypothesis_id="H005",
            technique="ctx-size reduction",
            model="LFM2-VL-450M",
            weight_dtype="q4_k_m",      # not a valid WeightDtype enum value
            runtime_backend="llamacpp_gguf",
        ))
        assert result["status"] == "error"

    def test_valid_proposal_returns_queued(self, tmp_path, monkeypatch):
        # Redirect queue writes to a temp dir
        import agents.search_strategist as ss
        monkeypatch.setattr(ss, "QUEUE_FILE", tmp_path / "queue.json")

        result = json.loads(_tool_propose_experiment(
            hypothesis_id="H005",
            technique="ctx-size reduction (4096→1024)",
            model="LFM2-VL-450M",
            rationale="Reduce KV-cache allocation to shrink memory footprint.",
            expected_gain="Mem -15%, TPS +5%",
            gain_axis="mem",
            weight_dtype="int4",
            runtime_backend="llamacpp_gguf",
            n_ctx=1024,
        ))
        assert result["status"] == "queued"
        assert "experiment_id" in result

    def test_valid_proposal_writes_queue_file(self, tmp_path, monkeypatch):
        import agents.search_strategist as ss
        queue_path = tmp_path / "queue.json"
        monkeypatch.setattr(ss, "QUEUE_FILE", queue_path)

        _tool_propose_experiment(
            hypothesis_id="H005",
            technique="ctx-size reduction",
            model="LFM2-VL-450M",
            weight_dtype="int4",
            runtime_backend="llamacpp_gguf",
        )
        assert queue_path.exists()
        queue = json.loads(queue_path.read_text())
        assert len(queue) == 1
        assert queue[0]["hypothesis_id"] == "H005"
