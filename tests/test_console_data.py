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


def test_default_log_path_picks_newest_then_falls_back(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    monkeypatch.setattr(cd, "RUN_LOG_DIR", logs)
    monkeypatch.setattr(cd, "_LEGACY_LOG", tmp_path / "legacy.log")
    assert cd.default_log_path() == ""              # nothing yet
    (tmp_path / "legacy.log").write_text("x")
    assert cd.default_log_path().endswith("legacy.log")
    logs.mkdir()
    import os, time
    (logs / "old.log").write_text("a")
    time.sleep(0.01)
    (logs / "new.log").write_text("b")
    os.utime(logs / "new.log", None)
    assert cd.default_log_path().endswith("new.log")  # newest wins over legacy


def test_local_server_up_false_on_closed_port():
    # a port nothing is listening on → False, fast (no hang)
    assert cd.local_server_up(port=59999, timeout=0.2) is False


def test_runlog_tee_writes_file(tmp_path, monkeypatch):
    from services import runlog
    monkeypatch.setattr(runlog, "RUN_LOG_DIR", tmp_path / "logs")
    with runlog.tee_stdout("myrun") as p:
        print("hello from the run")
    assert p.exists() and "hello from the run" in p.read_text()


def test_latest_proposals(tmp_path, monkeypatch):
    cq = tmp_path / "cq.json"
    cq.write_text(json.dumps([{"hypothesis_id": "P2-B1", "rationale": "build from 3B",
                               "proposed_at": "2026-06-15T09:00:00Z"}]))
    monkeypatch.setattr(cd, "CONSTRUCTION_QUEUE", cq)
    monkeypatch.setattr(cd, "EXPERIMENT_QUEUE", tmp_path / "missing.json")
    props = cd.latest_proposals()
    assert len(props) == 1 and props[0]["hypothesis"] == "P2-B1"
    assert props[0]["kind"] == "construction" and props[0]["rationale"] == "build from 3B"


def test_hypothesis_rows_includes_p2b1():
    rows = cd.hypothesis_rows()
    ids = {r["id"] for r in rows}
    assert "P2-B1" in ids and all("status" in r for r in rows)


def test_save_and_load_run_config_roundtrip(tmp_path):
    from schemas.run_config import RunConfig, load_run_config, save_run_config
    cfg = RunConfig(goal="match the benchmark", success_criteria={"POPE": 86.0},
                    allowed_hypotheses=["P2-B1"], chat_backend="local")
    p = save_run_config(cfg, tmp_path / "run.yaml")
    back = load_run_config(p)
    assert back.goal == "match the benchmark" and back.success_criteria["POPE"] == 86.0
    assert back.allowed_hypotheses == ["P2-B1"] and back.chat_backend == "local"
