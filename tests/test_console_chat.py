"""ADR-0013 H2b — operator chat dock backend (CI-safe; strategist mocked)."""

from __future__ import annotations

import sys
import types

import services.console_chat as cc


def _install_fake_strategist(monkeypatch, reply=None, raises=None):
    mod = types.ModuleType("agents.search_strategist")

    class FakeStrategist:
        def __init__(self, backend=None, verbose=True):
            self.backend = backend
            if raises:
                # raise at construction to simulate import/connect failure
                pass

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
