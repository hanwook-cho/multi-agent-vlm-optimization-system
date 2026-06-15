"""ADR-0013 H2 — operator console data layer (pure, CI-safe; no Streamlit)."""

from __future__ import annotations

import json

from services import console_data as cd


def test_queue_len(tmp_path):
    p = tmp_path / "q.json"
    assert cd.queue_len(p) == 0           # missing
    p.write_text(json.dumps([{"a": 1}, {"b": 2}]))
    assert cd.queue_len(p) == 2
    p.write_text("not json")
    assert cd.queue_len(p) == 0           # unreadable → 0


def test_backend_label_default_local_and_optins(tmp_path):
    assert cd.backend_label(env={}) == "local"
    assert cd.backend_label(env={"STRATEGIST_BACKEND": "api"}) == "api"
    assert cd.backend_label(env={"STRATEGIST_BACKEND": "anthropic"}) == "api"
    assert cd.backend_label(env={"STRATEGIST_BACKEND": "local"}) == "local"
    y = tmp_path / "run.yaml"
    y.write_text("chat_backend: api\n")
    assert cd.backend_label(run_yaml=y, env={}) == "api"


def test_parse_progress():
    log = ("loading...\n"
           "    [align] step 200/200  loss=2.381\n"
           "    [distill] step 639/1000  loss=0.29\n"
           "    [distill] step 640/1000  loss=0.74\n")
    p = cd.parse_progress(log)
    assert p == {"stage": "distill", "step": 640, "total": 1000, "loss": 0.74}
    assert cd.parse_progress("no steps here") is None


def test_log_tail(tmp_path):
    f = tmp_path / "run.log"
    f.write_text("\n".join(f"line {i}" for i in range(50)))
    out = cd.log_tail(f, n=5).splitlines()
    assert out == ["line 45", "line 46", "line 47", "line 48", "line 49"]
    assert cd.log_tail(tmp_path / "missing.log") == ""


def test_recent_constructions(tmp_path):
    led = tmp_path / "ledger"; led.mkdir()
    (led / "construction_abc.json").write_text(json.dumps({
        "experiment_id": "abcdef0123456789", "hypothesis_id": "P2-B1",
        "recorded_at": "2026-06-15T08:00:00Z",
        "report": {"status": "built", "quality_scores": [
            {"benchmark": "POPE", "metric": "Overall", "value": 0.0},
            {"benchmark": "MMBench_DEV_EN", "metric": "Overall", "value": 0.0}]},
    }))
    rows = cd.recent_constructions(led, n=8)
    assert len(rows) == 1
    r = rows[0]
    assert r["experiment_id"] == "abcdef012345" and r["hypothesis"] == "P2-B1"
    assert r["status"] == "built" and r["POPE"] == 0.0 and r["RealWorldQA"] is None


def test_pending_approvals_absent(tmp_path):
    assert cd.pending_approvals(tmp_path / "nope.json") == []
