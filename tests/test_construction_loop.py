"""B1.2 — the construction loop: agent proposes a StudentSpec → builder runs →
ledger → re-route. Tests the deterministic half (no LLM server, no heavy model).
"""

from __future__ import annotations

import json

import agents.search_strategist as ss
import services.construction_loop as cl
from schemas.students import StudentSpec


# ── propose_student tool (agent → construction queue) ───────────────────────

def test_propose_student_validates_and_queues(tmp_path, monkeypatch):
    q = tmp_path / "construction_queue.json"
    monkeypatch.setattr(ss, "CONSTRUCTION_QUEUE", q)
    out = json.loads(ss._tool_propose_student(
        hypothesis_id="P2-B1",
        lm="Qwen/Qwen2.5-0.5B-Instruct",
        vision="google/siglip-base-patch16-224",
        rationale="Both D-series regressed; build a right-sized student from the 3B lineage.",
        distill_data="qa_balanced_5k",
    ))
    assert out["status"] == "queued"
    queue = json.loads(q.read_text())
    assert len(queue) == 1
    entry = queue[0]
    assert entry["hypothesis_id"] == "P2-B1"
    # The queued spec round-trips through StudentSpec and is content-addressed.
    spec = StudentSpec.model_validate(entry["spec"])
    assert spec.content_hash() == entry["experiment_id"]
    assert spec.distill.data == "qa_balanced_5k"


def test_propose_student_errors_on_missing_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "CONSTRUCTION_QUEUE", tmp_path / "q.json")
    out = json.loads(ss._tool_propose_student(hypothesis_id="P2-B1", lm="", vision=""))
    assert out["status"] == "error" and "lm" in out["message"]


def test_propose_student_is_registered_as_a_tool():
    names = {t["name"] for t in ss.TOOLS}
    assert "propose_student" in names


# ── construction loop (queue → build → ledger) ──────────────────────────────

def _fake_build(spec, out_dir, smoke, **_kw):
    """Stand in for runners.build_student.build (no torch/model load)."""
    return {
        "experiment_id": spec.content_hash(),
        "status": "smoke_ok" if smoke else "built",
        "started_at": "2026-06-15T00:00:00+00:00",
        "completed_at": "2026-06-15T00:01:00+00:00",
        "align_losses": [2.2, 2.1],
        "distill_losses": [11.5, 10.8],
        "generations": [{"prompt": "Is there a cat?", "output": "..."}],
        "note": "stubbed",
    }


def test_run_once_writes_ledger_entry(tmp_path, monkeypatch):
    import runners.build_student as bs
    monkeypatch.setattr(bs, "build", _fake_build)
    ledger = tmp_path / "ledger"
    monkeypatch.setattr(cl, "LEDGER_DIR", ledger)

    spec = StudentSpec.model_validate_json(cl.DEFAULT_SPEC.read_text())
    entry = cl.run_once(spec, smoke=True, queue_entry={"hypothesis_id": "P2-B1", "rationale": "r"},
                        out_dir=tmp_path / "out")

    # ledger entry written, keyed by spec hash, with the construction shape
    files = list(ledger.glob("construction_*.json"))
    assert len(files) == 1
    written = json.loads(files[0].read_text())
    assert written["experiment_id"] == spec.content_hash()
    assert written["kind"] == "student_construction"
    assert written["hypothesis_id"] == "P2-B1"
    assert written["report"]["status"] == "smoke_ok"
    assert written["report"]["distill_final_loss"] == 10.8


def test_next_proposed_spec_prefers_queue_then_falls_back(tmp_path, monkeypatch):
    q = tmp_path / "cq.json"
    monkeypatch.setattr(cl, "CONSTRUCTION_QUEUE", q)
    # empty queue → default P2-B1 spec
    spec, entry = cl.next_proposed_spec()
    assert entry is None and spec.lm.startswith("Qwen/")
    # queued proposal → that spec is used
    proposed = StudentSpec.model_validate_json(cl.DEFAULT_SPEC.read_text())
    proposed = proposed.model_copy(update={"notes": "agent pick"})
    q.write_text(json.dumps([{"hypothesis_id": "P2-B1", "rationale": "x",
                              "experiment_id": proposed.content_hash(),
                              "spec": json.loads(proposed.model_dump_json())}]))
    spec2, entry2 = cl.next_proposed_spec()
    assert entry2 is not None and entry2["hypothesis_id"] == "P2-B1"
    assert spec2.content_hash() == proposed.content_hash()
