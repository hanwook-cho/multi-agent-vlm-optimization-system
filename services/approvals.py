"""
services/approvals.py
─────────────────────
The Human Approval Queue (HLD §6.1 #8, ADR-0013 H3) — the single source of truth
for gated decisions (deploy, eval-set change, Mode-B escalation, running a
constructed-student spec, promoting a technique).

One append-only-in-spirit JSON log: each request is a record whose status
transitions pending → approved/rejected, and the decision (who/when/note) is
recorded for audit. The operator console surfaces this one log three ways (global
bell, inline Monitor card, Approvals tab). Producers call request_approval(); the
console calls decide(); gated code can block on wait_for_approval().

CLI:
    python -m services.approvals list
    python -m services.approvals request deploy "Deploy student df64c49b to iPhone"
    python -m services.approvals approve <id> --note "looks good"
    python -m services.approvals reject  <id>
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
APPROVAL_LOG = PROJECT_ROOT / "artifacts" / "approval_log.json"

_VALID_DECISIONS = {"approve", "reject"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path = APPROVAL_LOG) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write(items: list[dict], path: Path = APPROVAL_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))


def request_approval(kind: str, summary: str, detail: dict | None = None,
                     by: str = "system", path: Path = APPROVAL_LOG) -> str:
    """Append a pending approval request. Returns its id."""
    items = _read(path)
    rec = {
        "id": uuid.uuid4().hex[:8],
        "kind": kind,
        "summary": summary,
        "detail": detail or {},
        "status": "pending",
        "requested_by": by,
        "requested_at": _now(),
        "decided_by": None,
        "decided_at": None,
        "note": "",
    }
    items.append(rec)
    _write(items, path)
    return rec["id"]


def list_all(path: Path = APPROVAL_LOG) -> list[dict]:
    return _read(path)


def list_pending(path: Path = APPROVAL_LOG) -> list[dict]:
    return [r for r in _read(path) if r.get("status") == "pending"]


def get(approval_id: str, path: Path = APPROVAL_LOG) -> dict | None:
    return next((r for r in _read(path) if r["id"] == approval_id), None)


def decide(approval_id: str, decision: str, by: str = "operator", note: str = "",
           path: Path = APPROVAL_LOG) -> dict:
    """Approve or reject a pending request. Idempotent-safe: errors if already decided."""
    if decision not in _VALID_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(_VALID_DECISIONS)}")
    items = _read(path)
    for r in items:
        if r["id"] == approval_id:
            if r["status"] != "pending":
                raise ValueError(f"approval {approval_id} already {r['status']}")
            r["status"] = "approved" if decision == "approve" else "rejected"
            r["decided_by"] = by
            r["decided_at"] = _now()
            r["note"] = note
            _write(items, path)
            return r
    raise KeyError(f"no approval with id {approval_id}")


def wait_for_approval(approval_id: str, poll: float = 5.0, sleep=time.sleep,
                      path: Path = APPROVAL_LOG) -> str:
    """Block until a request is decided; return 'approved'/'rejected'. For gated code."""
    while True:
        r = get(approval_id, path)
        if r is None:
            raise KeyError(f"no approval with id {approval_id}")
        if r["status"] != "pending":
            return r["status"]
        sleep(poll)


def main():
    ap = argparse.ArgumentParser(description="Human approval queue (ADR-0013 H3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    rq = sub.add_parser("request"); rq.add_argument("kind"); rq.add_argument("summary")
    ap_ = sub.add_parser("approve"); ap_.add_argument("id"); ap_.add_argument("--note", default="")
    rj = sub.add_parser("reject"); rj.add_argument("id"); rj.add_argument("--note", default="")
    args = ap.parse_args()

    if args.cmd == "list":
        for r in _read():
            mark = {"pending": "·", "approved": "✓", "rejected": "✗"}.get(r["status"], "?")
            print(f"  {mark} [{r['id']}] {r['kind']}: {r['summary']}  ({r['status']})")
    elif args.cmd == "request":
        print("queued:", request_approval(args.kind, args.summary, by="cli"))
    elif args.cmd == "approve":
        print("approved:", decide(args.id, "approve", note=args.note)["id"])
    elif args.cmd == "reject":
        print("rejected:", decide(args.id, "reject", note=args.note)["id"])


if __name__ == "__main__":
    main()
