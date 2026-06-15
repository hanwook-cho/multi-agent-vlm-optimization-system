"""ADR-0013 H3 — human approval queue (CI-safe; tmp log)."""

from __future__ import annotations

import pytest

from services import approvals as ap


def _log(tmp_path):
    return tmp_path / "approval_log.json"


def test_request_then_pending(tmp_path):
    p = _log(tmp_path)
    aid = ap.request_approval("deploy", "Deploy student df64c49b", {"device": "iphone"}, path=p)
    pend = ap.list_pending(p)
    assert len(pend) == 1 and pend[0]["id"] == aid
    assert pend[0]["status"] == "pending" and pend[0]["detail"]["device"] == "iphone"


def test_approve_moves_out_of_pending(tmp_path):
    p = _log(tmp_path)
    aid = ap.request_approval("escalate", "Mode B", path=p)
    r = ap.decide(aid, "approve", by="operator", note="ok", path=p)
    assert r["status"] == "approved" and r["decided_by"] == "operator" and r["note"] == "ok"
    assert ap.list_pending(p) == []
    assert len(ap.list_all(p)) == 1


def test_reject(tmp_path):
    p = _log(tmp_path)
    aid = ap.request_approval("eval_change", "swap eval set", path=p)
    assert ap.decide(aid, "reject", path=p)["status"] == "rejected"


def test_double_decide_errors(tmp_path):
    p = _log(tmp_path)
    aid = ap.request_approval("deploy", "x", path=p)
    ap.decide(aid, "approve", path=p)
    with pytest.raises(ValueError):
        ap.decide(aid, "reject", path=p)


def test_bad_decision_and_missing_id(tmp_path):
    p = _log(tmp_path)
    aid = ap.request_approval("deploy", "x", path=p)
    with pytest.raises(ValueError):
        ap.decide(aid, "maybe", path=p)
    with pytest.raises(KeyError):
        ap.decide("nope", "approve", path=p)


def test_wait_for_approval_blocks_then_returns(tmp_path):
    p = _log(tmp_path)
    aid = ap.request_approval("deploy", "x", path=p)
    calls = {"n": 0}

    def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] >= 2:
            ap.decide(aid, "approve", path=p)

    assert ap.wait_for_approval(aid, poll=0.0, sleep=fake_sleep, path=p) == "approved"
    assert calls["n"] >= 2


def test_missing_log_is_empty(tmp_path):
    assert ap.list_all(tmp_path / "none.json") == []
    assert ap.list_pending(tmp_path / "none.json") == []
