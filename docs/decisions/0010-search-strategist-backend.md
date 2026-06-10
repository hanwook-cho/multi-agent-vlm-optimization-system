# ADR-0010: Search Strategist LLM Backend Selection

**Date:** 2026-06-05
**Status:** Accepted
**Context:** Task 5.1 — `agents/search_strategist.py` backend architecture

---

## Context

The Search Strategist is an LLM agent that reads the experiment ledger and Pareto frontier, then proposes the next experiment via tool calls. It needs an LLM backend.

Three options were considered:
1. **Anthropic API only** — simplest implementation, best model quality, requires API key and network
2. **OpenAI-compatible only** — works with any model behind an OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, etc.), no Anthropic dependency
3. **Multi-backend with auto-detection** — supports both, selects at runtime, degrades gracefully

The additional question was whether a local LLM (gemma3 via Ollama) is capable enough to run the agent without cloud dependency. This was tested empirically before the decision was finalised.

---

## Decision

**Implement a multi-backend architecture with auto-detection and a ReAct JSON fallback.**

- `SearchStrategist(backend="anthropic"|"ollama"|"openai_compat"|"auto", model=..., base_url=...)`
- `_AnthropicBackend`: uses `anthropic.Anthropic()`, Anthropic tool-use format
- `_OpenAICompatibleBackend`: uses `openai.OpenAI()`, OpenAI tool schema format
  - Attempts native tool-calling first (`tools=`, `tool_choice="auto"`)
  - On HTTP 400 "does not support tools": switches to ReAct JSON mode automatically
- `backend="auto"`: tries Anthropic SDK if available, falls back to OpenAI-compatible

Tool descriptions are maintained in Anthropic's `input_schema` format (source of truth); a conversion function `_openai_tools_from_anthropic()` produces the OpenAI `function.parameters` form. This keeps a single canonical tool definition regardless of backend.

---

## Rationale

### Why not Anthropic-only

Development and offline testing require no API key or network. A local Ollama backend covers these cases at zero marginal cost. The ReAct fallback also makes the agent usable on any model that can follow JSON instructions — including models with no native tool-calling API.

### Why not OpenAI-compatible-only

Claude-3.5-sonnet/haiku is measurably better at the strategic reasoning this agent requires (see Observation `2026-06-05-gemma3-react-evaluation.md`). Dropping Anthropic support would degrade production quality for no benefit.

### Why ReAct JSON fallback

Many models exposed via Ollama (gemma3, mistral, phi-4) don't implement the `/v1/chat/completions` `tools` parameter. Without a fallback, the agent crashes on first call. The ReAct pattern (Reason → Act → Observe loop via structured JSON in plain chat) is well-established and has near-universal model compatibility. The cost is a less reliable format (regex parsing vs. structured tool dispatch) and higher round-trip count.

### Why tool schema source-of-truth in Anthropic format

Anthropic's `input_schema` is a strict JSON Schema subset. The OpenAI format is a superset with a `function` wrapper. Converting Anthropic → OpenAI is lossless; the reverse is not always true. Keeping the source in the stricter format prevents schema drift.

---

## Consequences

**Commits to:**
- Maintaining two schema representations (Anthropic + OpenAI) in sync via `_openai_tools_from_anthropic()`
- Testing ReAct path separately from native tool-call path when adding new tools
- Versioning tools: any new tool added to `TOOLS` must be reflected in both backends

**Forecloses:**
- Anthropic-specific features that have no OpenAI equivalent (e.g. computer use, prompt caching metadata) — acceptable for this agent's use case
- Stateful multi-turn tool use with native Anthropic format when running in ReAct mode (ReAct is always single-turn per round)

**Performance note:**
- ReAct mode uses ~1.5–2× more rounds than native tool-call mode (redundant query steps)
- At Phase 1 scope (6 rounds, 7 hypotheses) this is acceptable
- At Phase 2 scope (longer ledger, more hypotheses) consider increasing `max_rounds` for ReAct backends or filtering the hypothesis table before injection

---

## Alternatives considered

### Fine-tuned local model

Training a small model specifically for the tool-call format would eliminate ReAct round overhead. Rejected: requires labelled training data we don't have, maintenance burden, and the quality ceiling is below the cloud models anyway.

### Prompt-only (no tools)

Have the LLM output a raw `ExperimentConfig` JSON directly without tool calls. Simpler, but loses the inspect-before-propose loop that catches already-tried hypotheses. The `query_results` tool call is what gives the agent access to live ledger state; without it the agent works from its context window only and cannot self-correct.

### LangChain / LlamaIndex agent framework

Rejected in favour of a thin hand-rolled implementation. The framework would add a dependency, abstract the ReAct loop in ways that make debugging harder, and is unlikely to provide meaningful value for a 3-tool agent.

---

## Addendum (2026-06-08): Local model + serving choice

After evaluating gemma3:4b (see Observation `2026-06-05-gemma3-react-evaluation.md`), the local backend was revised:

**Serving:** llama.cpp's `llama-server` with `--jinja`, not Ollama. Rationale:
- `--jinja` enables the model's chat template, so the OpenAI `tools` parameter is honored natively. This eliminates the ReAct fallback for tool-trained models and the round-waste that comes with it.
- llama.cpp also supports JSON-schema / GBNF constrained decoding (`response_format`), which can force schema-valid output from *any* model — kept in reserve, not needed for Qwen2.5 since it is tool-trained.
- The 400-on-tools error that forced ReAct mode was an Ollama-side template gate, not a model limitation.

**Model:** Qwen2.5-Instruct, not Gemma. Rationale:
- Best-in-class native tool-calling at the 7B weight class; stronger structured-output and instruction-following than same-size Gemma.
- Gemma (any size, including 27B) lacks native tool-calling via these servers and is stuck on ReAct.

**Model sizing by hardware:**
| Hardware | Default model | Footprint (Q4_K_M) |
|---|---|---|
| M4 16GB Mac mini (current) | `qwen2.5-7b-instruct` | ~5GB runtime; leaves room for tracker/dashboard/DB |
| M5-class 32GB (future) | `qwen2.5-32b-instruct` | ~19–20GB; large reasoning jump, latency irrelevant for this role |
| 32GB, recall-bound | `deepseek-r1-distill-qwen-32b` | experimental — reasoning-distilled, attacks strategic-recall gap |

The Search Strategist is **not latency-critical** (experiments take minutes-to-hours; proposal latency is noise), so larger/slower models are acceptable when memory allows.

**Phase applicability:** Qwen2.5 (local) powers the Search Strategist through Phases 1–3+ (Mode A runs continuously per HLD §4.3). Phase 3 *adds* a frontier-API Research Analyst (Mode B) alongside it — it does not replace the local strategist. Per HLD §7.2, the Research Analyst must use a frontier API (citation-hallucination risk on dense papers), not a local model.

**Code:** `backend="llamacpp"` is the new local default (also the fallback for `backend="auto"` when no `ANTHROPIC_API_KEY`). Defaults: `http://localhost:8080/v1`, `qwen2.5-7b-instruct`. Launch via `scripts/start_strategist_llm.sh`. Ollama remains supported via `backend="ollama"` (gemma3, ReAct fallback) for memory-constrained dev.

---

## Open issues

- [x] ~~gemma3 re-proposes already-completed hypotheses despite `status: completed` in the table.~~ Fixed: `_build_system_prompt()` now accepts an optional `hypothesis_table` parameter and splits the table into "Open (NOT_TRIED)" and "Closed (CONFIRMED/NULL_RESULT/BLOCKED)" sections. Only open hypotheses are shown as candidates; closed ones appear as a read-only summary. Covered by `TestBuildSystemPrompt` in `tests/test_search_strategist.py`.
- [x] ~~`[gemma3 raw]` debug print in `_chat_react()` should be gated behind `self.verbose`.~~ Fixed: renamed to `[backend raw]` (model-agnostic), gated on `self.verbose`. `verbose` parameter threaded through `SearchStrategist` → `_OpenAICompatibleBackend`.
- [x] ~~No unit tests for `_openai_tools_from_anthropic()` or the ReAct parser.~~ Fixed: 26 tests added in `tests/test_search_strategist.py` covering schema conversion, ReAct parser (including documented bare-JSON nested-brace limitation), prompt filtering, and `_tool_propose_experiment` validation.
