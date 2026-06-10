# Observation: gemma3:4b as Search Strategist Backend via Ollama ReAct

**Date:** 2026-06-05
**Context:** Phase 1 exit — verifying local-LLM viability for `agents/search_strategist.py`
**Trigger:** Question: "Any reason to use Anthropic API? No local LLM such as gemma?"

> **Update (2026-06-10):** Two findings below are now superseded:
> 1. The **H002 re-proposal bug is fixed** — `_build_system_prompt()` filters
>    completed/blocked hypotheses out of the candidate list (commit + tests this session).
> 2. **Qwen2.5 is no longer "not tested."** Qwen2.5-7B-Instruct via llama.cpp was
>    verified live and beat gemma3 on every axis (native tool-calling, no completed-hypothesis
>    re-proposal, converged in 4 rounds vs 6). It is now the **default local backend**.
>    See [`2026-06-10-qwen25-llamacpp-verification.md`](2026-06-10-qwen25-llamacpp-verification.md)
>    and the ADR-0010 addendum. This doc is retained as the historical gemma3 record.

---

## Setup

- **Model:** `gemma3:4b` via Ollama (local, CPU+GPU, no cloud)
- **Endpoint:** `http://localhost:11434/v1` (OpenAI-compatible)
- **Mode:** ReAct JSON fallback (gemma3 returns HTTP 400 on `/v1/chat/completions` with `tools=`)
- **Rounds:** 6 max
- **Hypothesis table:** H001–H007, statuses populated from actual ledger

---

## What worked

| Behaviour | Detail |
|---|---|
| Auto-detect 400 error | Caught "does not support tools" on first call; switched to ReAct JSON mode silently without crashing |
| Format adherence | Consistently wrapped output in ` ```json {...} ``` ` every round — regex parser never failed |
| Tool sequencing | Called `query_results` before `propose_experiment` (correct per policy) |
| Schema self-correction | Round 5: used `"q4_k_m"` as `weight_dtype`; received validation error JSON back; adjusted to `"int4"` on retry |
| Valid final config | Produced an `ExperimentConfig` that passes Pydantic validation end-to-end |

## Where it fell short

| Issue | Detail |
|---|---|
| Re-proposed H002 | H002 is already `completed` in the ledger; a stronger model would have skipped it and proposed H005 or H006 |
| Round budget exhausted | Used all 6/6 rounds — wandered between `query_results` and `query_frontier` before converging |
| Shallow grounding | Rationale ("Reduce ttft_ms by 5–10ms") was plausible but not derived from ledger numbers |

---

## Root cause of the H002 re-proposal

The system prompt lists H002 as `status: completed`. gemma3 read this, but its strategic reasoning didn't reliably exclude completed hypotheses from consideration. The model treated "completed" as a data point about the technique rather than as a blocker on that specific hypothesis ID.

A mitigation: filter `status: completed` hypotheses out of the table entirely before injecting it into the prompt, so the model never sees them as options.

---

## Verdict

**gemma3:4b is a viable orchestrator for the Search Strategist at Phase 1 scope.**

The ReAct JSON pattern works reliably. The model handles tool call/response cycles, reads structured JSON back from tools, and self-corrects on schema validation errors. For a 7-item hypothesis table it gets to a valid proposal every run.

The gap vs. claude-sonnet is **strategic recall**: gemma3 doesn't track "already tried" as carefully, and exhausts its round budget doing redundant queries. As the hypothesis table grows (Phase 2: 15+ entries, longer ledger), this gap will widen.

**Superseded escalation path (see Update banner).** The conclusion changed after testing Qwen2.5 and switching serving to llama.cpp. Current recommendation (ADR-0010 addendum):

- M4 16GB (current): `qwen2.5-7b-instruct` via **llama.cpp `--jinja`** — native tool-calling, verified, default
- 32GB (future Mac): `qwen2.5-32b-instruct`; `deepseek-r1-distill-qwen-32b` if recall-bound
- gemma3 retained only as an Ollama fallback for memory-constrained dev

---

## Code changes made during this evaluation

1. **`_OpenAICompatibleBackend._chat_tools()`** — catches HTTP 400 "does not support tools", rebuilds system prompt with ReAct suffix, falls through to `_chat_react()`
2. **`_chat_react()`** — parses ` ```json {...} ``` ` blocks; fallback regex for bare JSON objects; returns tool name + parsed args
3. **`_tool_propose_experiment()`** — all args made optional with defaults; `**_extra` added to absorb unexpected keys; returns structured error JSON on missing required fields instead of crashing
4. **`results.append()`** — added `"name": tool_name` to result dict (was missing in ReAct path, causing KeyError downstream)

All changes committed in: `agents/search_strategist.py: ReAct JSON fallback for tool-less models`
