"""ADR-0013 H1 — operator run controls + run.yaml intake (CI-safe, no model)."""

from __future__ import annotations

import pytest

import services.run_control as rc
from schemas.run_config import RunConfig, load_run_config


@pytest.fixture(autouse=True)
def _isolated_control(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "CONTROL_FILE", tmp_path / "run_control.json")


def test_default_is_run():
    assert rc.get_state()["state"] == "run"
    rc.check()  # no raise


def test_stop_raises_graceful():
    rc.set_state("stop", "good enough")
    with pytest.raises(rc.RunStopped) as ei:
        rc.check()
    assert ei.value.mode == "stop" and ei.value.reason == "good enough"


def test_kill_raises():
    rc.set_state("kill")
    with pytest.raises(rc.RunStopped) as ei:
        rc.check()
    assert ei.value.mode == "kill"


def test_resume_clears_to_run():
    rc.set_state("stop")
    rc.set_state("resume")
    assert rc.get_state()["state"] == "run"
    rc.check()


def test_clear():
    rc.set_state("pause")
    rc.clear()
    assert rc.get_state()["state"] == "run"


def test_invalid_state_rejected():
    with pytest.raises(ValueError):
        rc.set_state("explode")


def test_wait_if_paused_blocks_then_resumes():
    rc.set_state("pause")
    calls = {"n": 0}

    def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] >= 2:
            rc.set_state("run")  # operator resumes after a couple polls

    rc.wait_if_paused(poll=0.0, sleep=fake_sleep)
    assert calls["n"] >= 2 and rc.get_state()["state"] == "run"


def test_pause_then_kill_is_honored_on_resume():
    rc.set_state("pause")

    def fake_sleep(_):
        rc.set_state("kill", "swap thrash")

    with pytest.raises(rc.RunStopped) as ei:
        rc.wait_if_paused(poll=0.0, sleep=fake_sleep)
    assert ei.value.mode == "kill"


# ── run.yaml intake ──────────────────────────────────────────────────────────

def test_run_config_defaults():
    c = RunConfig(goal="match the benchmark")
    assert c.target_device == "mac_mini_m4_16gb"
    assert c.eval_set == ["POPE", "RealWorldQA", "MMBench_DEV_EN"]
    assert c.allowed_hypotheses == []


def test_load_run_config_roundtrip(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(
        "goal: build a small student\n"
        "success_criteria: {POPE: 86.0}\n"
        "allowed_hypotheses: [P2-B1]\n"
    )
    c = load_run_config(p)
    assert c.goal == "build a small student"
    assert c.success_criteria["POPE"] == 86.0
    assert c.allowed_hypotheses == ["P2-B1"]


def test_example_run_yaml_is_valid():
    from pathlib import Path
    root = Path(__file__).parent.parent
    c = load_run_config(root / "configs" / "run.example.yaml")
    assert "P2-B1" in c.allowed_hypotheses and c.success_criteria["POPE"] == 86.0


def test_run_config_chat_backend_defaults_local():
    assert RunConfig(goal="x").chat_backend == "local"
    with pytest.raises(ValueError):
        RunConfig(goal="x", chat_backend="cloud")  # only local|api allowed


# ── agent backend resolution: default local, API opt-in (ADR-0013) ──────────

def test_backend_defaults_to_local_even_with_api_key_present():
    from agents.search_strategist import _resolve_backend_name
    # ANTHROPIC_API_KEY present but no explicit opt-in → still local
    env = {"ANTHROPIC_API_KEY": "sk-xxx"}
    assert _resolve_backend_name("auto", env=env) == "llamacpp"


def test_backend_env_opt_in_to_api():
    from agents.search_strategist import _resolve_backend_name
    assert _resolve_backend_name("auto", env={"STRATEGIST_BACKEND": "api"}) == "anthropic"
    assert _resolve_backend_name("auto", env={"STRATEGIST_BACKEND": "local"}) == "llamacpp"


def test_backend_explicit_arg_and_aliases():
    from agents.search_strategist import _resolve_backend_name
    assert _resolve_backend_name("local") == "llamacpp"
    assert _resolve_backend_name("api") == "anthropic"
    assert _resolve_backend_name("ollama", env={}) == "ollama"
