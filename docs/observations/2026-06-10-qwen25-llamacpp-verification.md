# Observation: Qwen2.5-7B-Instruct via llama.cpp as Search Strategist Backend

**Date:** 2026-06-10
**Context:** Selecting the production local backend for `agents/search_strategist.py`
**Trigger:** "Which model is better if llama.cpp used?" → committed to Qwen2.5 + llama.cpp, then live-verified.
**Supersedes:** the gemma3 verdict in [`2026-06-05-gemma3-react-evaluation.md`](2026-06-05-gemma3-react-evaluation.md)

---

## Setup

- **Model:** `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (4.68GB, bartowski build)
- **Server:** `llama-server` built from `vendor/llama.cpp` with `-DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON`
- **Launch:** `scripts/start_strategist_llm.sh` — `--jinja` enabled (native tool-calling)
- **Endpoint:** `http://localhost:8080/v1` (`backend="llamacpp"`)
- **Hardware:** Mac mini M4 16GB. Model loaded in ~1s; healthy on `:8080`.
- **Hypothesis table:** H001–H007, real ledger statuses, completed-filter active.

---

## Result

Full agent loop ran end-to-end and produced a valid proposal:

| Check | Result |
|---|---|
| `backend="llamacpp"` connects to llama-server | ✅ |
| Native tool-calling (not ReAct) | ✅ **0** ReAct fallbacks — used `_chat_tools` throughout |
| Parallel multi-tool use | ✅ Round 1: `query_frontier` + `query_results`×2 + `propose_experiment` |
| Completed-hypothesis filter respected | ✅ Proposed only **H005/H006** (open); never re-proposed H001/H002/H003/H004 |
| Valid `ExperimentConfig` output | ✅ Final **H006 (GPTQ INT4 LM backbone)** passes Pydantic |
| Converged | ✅ 4 rounds (vs gemma3's 6/6) |

Final proposal: **H006 — GPTQ INT4 LM backbone on LFM2-VL-450M**, gain axis `quality`, with a coherent ledger-grounded rationale. A sound, untried choice.

---

## Qwen2.5-7B vs gemma3:4b

| | gemma3:4b (Ollama, ReAct) | Qwen2.5-7B (llama.cpp, native) |
|---|---|---|
| Tool mode | ReAct JSON fallback | **Native tool-calling** |
| Re-proposed a completed hypothesis? | Yes (H002) | **No** |
| Rounds to converge | 6/6 (exhausted) | **4** |
| Final proposal | H002 (already done) | **H006 (valid, untried)** |

Two changes drove the improvement: the completed-hypothesis filter (model-agnostic) and the model+serving swap (Qwen2.5 native tools via `--jinja`). The ReAct round-tax is gone entirely.

---

## Verdict

**Qwen2.5-7B-Instruct via llama.cpp is the production local backend for the Search Strategist** on the M4 16GB Mac mini. It is sufficient for Phase 2 (Search Strategist only; Research Analyst / frontier API arrives in Phase 3 per HLD §7.2). Sizing path for the future 32GB Mac is in the ADR-0010 addendum.

## Reproduce

```bash
# build (one-time)
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON
cmake --build vendor/llama.cpp/build --target llama-server -j

# model (one-time, ~4.7GB)
hf download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir ./models

# serve + run
LLAMA_SERVER=vendor/llama.cpp/build/bin/llama-server \
MODEL=models/Qwen2.5-7B-Instruct-Q4_K_M.gguf ./scripts/start_strategist_llm.sh &
python3 agents/search_strategist.py llamacpp
```

Both `llama-server` and the GGUF are gitignored (live in `vendor/llama.cpp/build/` and `models/`).
