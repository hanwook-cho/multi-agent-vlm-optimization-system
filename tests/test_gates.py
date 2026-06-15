"""HLD §5.1 / ADR-0013 — reusable human gates (CI-safe; approvals mocked)."""

from __future__ import annotations

from services import gates


def _capture(monkeypatch):
    seen = {}

    def fake_request(**kw):
        seen.update(kw)
        return "gid1"

    monkeypatch.setattr(gates.approvals, "request_approval", fake_request)
    return seen


def test_gated_block_returns_decision(monkeypatch):
    _capture(monkeypatch)
    monkeypatch.setattr(gates.approvals, "wait_for_approval", lambda *a, **k: "approved")
    assert gates.gated("deploy", "x", block=True) == "approved"


def test_gated_nonblock_returns_id(monkeypatch):
    seen = _capture(monkeypatch)
    out = gates.gated("mode_b_escalation", "x", block=False)
    assert out == "gid1" and seen["kind"] == "mode_b_escalation"


def test_gate_deploy_shape(monkeypatch):
    seen = _capture(monkeypatch)
    monkeypatch.setattr(gates.approvals, "wait_for_approval", lambda *a, **k: "rejected")
    assert gates.gate_deploy("LFM2-VL-450M-distill", "iphone16pro-001") == "rejected"
    assert seen["kind"] == "deploy"
    assert seen["detail"]["model"] == "LFM2-VL-450M-distill"
    assert seen["detail"]["device"] == "iphone16pro-001"


def test_gate_eval_change_shape(monkeypatch):
    seen = _capture(monkeypatch)
    monkeypatch.setattr(gates.approvals, "wait_for_approval", lambda *a, **k: "approved")
    gates.gate_eval_change(["POPE"], ["POPE", "RealWorldQA"])
    assert seen["kind"] == "eval_change" and seen["detail"]["new"] == ["POPE", "RealWorldQA"]


def test_gate_mode_b_escalation_is_nonblocking_by_default(monkeypatch):
    seen = _capture(monkeypatch)
    # wait_for_approval must NOT be called when non-blocking
    monkeypatch.setattr(gates.approvals, "wait_for_approval",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not block")))
    out = gates.gate_mode_b_escalation(dossier_path="artifacts/dossiers/d.md")
    assert out == "gid1" and seen["kind"] == "mode_b_escalation"
    assert seen["detail"]["dossier"] == "artifacts/dossiers/d.md"
