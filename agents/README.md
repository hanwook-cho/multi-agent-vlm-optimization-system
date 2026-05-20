# Agents

LLM-driven components that require sandboxing, decision logs, and hallucination guards.
See `docs/HLD.md §6.1` for the agent/service distinction.

**Search Strategist Agent** (`search_strategist.py`) — Phase 1
Proposes which configurations to try next in the Mode A search space.
Triages failed runs. Writes human-readable rationales for Pareto candidates.
Runs continuously, one decision per completed experiment batch.
Uses a local LLM (Qwen3-Coder-7B Q4 on Agent Mac).

**Research Analyst Agent** (`research_analyst.py`) — Phase 3
Ingests papers and repositories, extracts HypothesisRecord objects.
Runs weekly. Uses a frontier API (Claude or GPT-class) for its reasoning.
Output enters the Human Approval Queue; never acts autonomously on its own output.

Every agent action that changes system state is logged as an `AgentDecision`
(see `schemas/agents.py`).
