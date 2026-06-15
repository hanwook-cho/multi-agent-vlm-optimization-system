"""
services/console_chat.py
────────────────────────
Operator chat dock backend (ADR-0013 H2b). Thin wrapper around the Search
Strategist's plain-text chat, with graceful handling when the local llama-server
(or the API) is unavailable — the console must never crash because the LLM is down.

Backend is configurable, default local (ADR-0013): the operator talks to the local
Qwen2.5-7B unless they opted into the API.
"""

from __future__ import annotations


def chat_reply(message: str, history: list[dict] | None = None,
               backend: str | None = None) -> str:
    """Return the strategist's reply, or a friendly offline message on failure."""
    try:
        from agents.search_strategist import SearchStrategist
        agent = SearchStrategist(backend=backend or "local", verbose=False)
        reply = agent.chat(message, history=history)
        return reply or "(no reply)"
    except Exception as exc:
        return (f"(strategist unavailable — {type(exc).__name__}. "
                "Start the local model: scripts/start_strategist_llm.sh, "
                "or switch the backend to api.)")
