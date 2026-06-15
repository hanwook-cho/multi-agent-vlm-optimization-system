"""ADR-0013 H2b — operator chat dock backend (CI-safe; strategist mocked)."""

from __future__ import annotations

import sys
import types

import services.console_chat as cc


_CAPTURED = {}


def _install_fake_strategist(monkeypatch, reply=None, raises=None):
    mod = types.ModuleType("agents.search_strategist")

    class FakeStrategist:
        def __init__(self, **kwargs):
            _CAPTURED.clear()
            _CAPTURED.update(kwargs)

        def chat(self, message, history=None):
            if raises:
                raise raises
            return reply

    mod.SearchStrategist = FakeStrategist
    monkeypatch.setitem(sys.modules, "agents.search_strategist", mod)


def test_chat_reply_returns_strategist_text(monkeypatch):
    _install_fake_strategist(monkeypatch, reply="P2-B1 needs more alignment.")
    out = cc.chat_reply("why is the run stuck?", history=[], backend="local")
    assert out == "P2-B1 needs more alignment."


def test_chat_reply_handles_offline_gracefully(monkeypatch):
    _install_fake_strategist(monkeypatch, raises=ConnectionError("refused"))
    out = cc.chat_reply("status?", backend="local")
    assert "unavailable" in out.lower() and "ConnectionError" in out


def test_chat_reply_empty_reply_falls_back(monkeypatch):
    _install_fake_strategist(monkeypatch, reply="")
    assert cc.chat_reply("hi") == "(no reply)"


def test_chat_reply_forwards_api_credentials(monkeypatch):
    _install_fake_strategist(monkeypatch, reply="ok")
    cc.chat_reply("hi", backend="openai_compat", api_key="sk-test",
                  base_url="https://x/v1", model="gpt-x")
    assert _CAPTURED["backend"] == "openai_compat"
    assert _CAPTURED["api_key"] == "sk-test"
    assert _CAPTURED["base_url"] == "https://x/v1"
    assert _CAPTURED["model"] == "gpt-x"


def test_chat_reply_anthropic_key_falls_back_to_env(monkeypatch):
    _install_fake_strategist(monkeypatch, reply="ok")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    cc.chat_reply("hi", backend="anthropic")  # no key passed → env fallback
    assert _CAPTURED["api_key"] == "sk-env"
