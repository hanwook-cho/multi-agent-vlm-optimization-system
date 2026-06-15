"""
services/console_chat.py
────────────────────────
Operator chat dock backend (ADR-0013 H2b). Thin wrapper around the Search
Strategist's plain-text chat, with graceful handling when the local llama-server
(or the API) is unavailable — the console must never crash because the LLM is down.

Backend is configurable, default local (ADR-0013): the operator talks to the local
Qwen2.5-7B unless they opt into an API. API credentials (key/base-url/model) are
passed through from the UI and are NOT persisted by this module.
"""

from __future__ import annotations

import os


def chat_reply(message: str, history: list[dict] | None = None,
               backend: str | None = None, api_key: str | None = None,
               base_url: str | None = None, model: str | None = None) -> str:
    """Return the strategist's reply, or a friendly message on failure.

    backend: 'local' | 'anthropic' | 'openai_compat' | 'api' (alias). For API
    backends, api_key falls back to the usual env vars when not given in the UI.
    """
    try:
        from agents.search_strategist import SearchStrategist
        if backend in ("anthropic", "api") and not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        kwargs = {"backend": backend or "local", "verbose": False}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if model:
            kwargs["model"] = model
        agent = SearchStrategist(**kwargs)
        reply = agent.chat(message, history=history)
        return reply or "(no reply)"
    except Exception as exc:
        return (f"(strategist unavailable — {type(exc).__name__}: {exc}. "
                "For local: start scripts/start_strategist_llm.sh. "
                "For api: check the model, base URL, and key.)")
