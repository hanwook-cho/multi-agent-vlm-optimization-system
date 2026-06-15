"""
services/gates.py
─────────────────
Reusable human gates (HLD §5.1, ADR-0013 H3). Every always-gated decision — deploy,
eval-set change, Mode-B escalation, running a large-compute spec — funnels through
one helper that posts to the approval queue and (optionally) blocks until the
operator decides in the console. One pattern, one audit log.

    from services import gates
    if gates.gate_deploy("LFM2-VL-450M-distill", "iphone16pro-001") == "approved":
        ... deploy ...

CLI (blocks until decided, for manual/automated gating):
    python -m services.gates deploy "Deploy student df64c49b to iphone16pro-001"
    python -m services.gates mode_b_escalation "Mode A stalled — escalate" --no-block
"""

from __future__ import annotations

import argparse
import time

from services import approvals

GATE_KINDS = ("deploy", "eval_change", "mode_b_escalation", "construction_run")


def gated(kind: str, summary: str, detail: dict | None = None, by: str = "system",
          block: bool = True, poll: float = 5.0, sleep=time.sleep) -> str:
    """Post a gated action to the approval queue.

    block=True  → wait for the decision; return 'approved' or 'rejected'.
    block=False → return the approval id (the dossier-style "post and move on" case).
    """
    aid = approvals.request_approval(kind=kind, summary=summary, detail=detail or {}, by=by)
    print(f"  ⏸ approval requested (id {aid}, {kind}) — decide in the console's "
          f"Approvals tab, or: python -m services.approvals approve {aid}")
    if not block:
        return aid
    decision = approvals.wait_for_approval(aid, poll=poll, sleep=sleep)
    print(f"  {'✓ approved' if decision == 'approved' else '✗ ' + decision} — {kind}")
    return decision


def gate_deploy(model: str, device: str, block: bool = True, **detail) -> str:
    return gated("deploy", f"Deploy {model} to {device}",
                 {"model": model, "device": device, **detail}, by="deploy", block=block)


def gate_eval_change(old, new, block: bool = True) -> str:
    return gated("eval_change", f"Change eval set {old} → {new}",
                 {"old": old, "new": new}, by="eval", block=block)


def gate_mode_b_escalation(dossier_path=None, block: bool = False, **detail) -> str:
    """Escalate Mode A → Mode B. Default non-blocking: the dossier posts to the queue
    and the operator's approval IS the escalation decision (HLD §4.2)."""
    return gated("mode_b_escalation", "Escalate Mode A → Mode B (research-driven)",
                 {"dossier": str(dossier_path) if dossier_path else None, **detail},
                 by="threshold_monitor", block=block)


def main():
    ap = argparse.ArgumentParser(description="Human gates (HLD §5.1)")
    ap.add_argument("kind", choices=GATE_KINDS)
    ap.add_argument("summary")
    ap.add_argument("--no-block", action="store_true", help="Post and return immediately")
    args = ap.parse_args()
    out = gated(args.kind, args.summary, by="cli", block=not args.no_block)
    print(out)


if __name__ == "__main__":
    main()
