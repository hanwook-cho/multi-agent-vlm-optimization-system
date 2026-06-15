"""
services/construction_loop.py
─────────────────────────────
Phase 2 / ADR-0012 B1.2 — close the construction loop.

The Search Strategist proposes a StudentSpec (agents.search_strategist.propose_student
→ artifacts/construction_queue.json). This loop consumes the next proposal, runs the
generic builder (runners/build_student.build), and writes the result into the
experiment ledger keyed by the spec's content hash — so the strategist's next
propose_next() reads it and re-routes. That is the agent → spec → build → ledger →
re-route cycle the project is built around, for model CONSTRUCTION.

This module deliberately does NOT require the LLM server: it consumes whatever the
agent queued (or a default spec), so the deterministic half of the loop is testable
and runnable headless. The agent half (propose_student) feeds the same queue.

Usage
-----
    # build the next agent-proposed spec (smoke), record to the ledger
    python services/construction_loop.py --smoke

    # build a specific spec file
    python services/construction_loop.py --spec tests/fixtures/student_spec_p2b1_qwen05b_siglip.json --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.students import StudentSpec

CONSTRUCTION_QUEUE = PROJECT_ROOT / "artifacts" / "construction_queue.json"
LEDGER_DIR = PROJECT_ROOT / "artifacts" / "experiment_ledger"
DEFAULT_SPEC = PROJECT_ROOT / "tests" / "fixtures" / "student_spec_p2b1_qwen05b_siglip.json"


def next_proposed_spec() -> tuple[StudentSpec, dict | None]:
    """Pop nothing — peek the most recent queued construction proposal.

    Returns (spec, queue_entry). Falls back to the default P2-B1 spec when the
    queue is empty (so the loop is runnable before the agent has proposed).
    """
    if CONSTRUCTION_QUEUE.exists():
        try:
            queue = json.loads(CONSTRUCTION_QUEUE.read_text())
        except Exception:
            queue = []
        if queue:
            entry = queue[-1]
            return StudentSpec.model_validate(entry["spec"]), entry
    return StudentSpec.model_validate_json(DEFAULT_SPEC.read_text()), None


def _ledger_entry(spec: StudentSpec, record: dict, queue_entry: dict | None) -> dict:
    """Shape a build_record into a ledger entry the strategist's query tools read."""
    return {
        "experiment_id": spec.content_hash(),
        "kind": "student_construction",
        "hypothesis_id": (queue_entry or {}).get("hypothesis_id", "P2-B1"),
        "rationale": (queue_entry or {}).get("rationale", ""),
        "spec": json.loads(spec.model_dump_json()),
        "report": {
            "experiment_id": spec.content_hash(),
            "status": record.get("status"),
            "device_id": spec.target_device_id,
            "started_at": record.get("started_at"),
            "completed_at": record.get("completed_at"),
            "align_final_loss": (record.get("align_losses") or [None])[-1],
            "distill_final_loss": (record.get("distill_losses") or [None])[-1],
            "quality_scores": record.get("quality_scores", []),  # populated at B1.3 (real eval)
            "note": record.get("note", ""),
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def run_once(spec: StudentSpec, smoke: bool, queue_entry: dict | None = None,
             out_dir: Path | None = None, eval_after: bool = False,
             eval_n: int | None = None, align_steps: int | None = None,
             distill_steps: int | None = None, max_samples: int | None = None,
             require_approval: bool = False) -> dict:
    """Build the spec, optionally run the same-path eval, write a ledger entry.

    require_approval gates a REAL (non-smoke) run behind the operator approval queue
    (HLD §5.1, large compute): the run requests approval and BLOCKS until the operator
    approves/rejects in the console (or via `python -m services.approvals`).
    """
    exp = spec.content_hash()

    if require_approval and not smoke:
        from services import approvals
        hyp = (queue_entry or {}).get("hypothesis_id", "P2-B1")
        aid = approvals.request_approval(
            kind="construction_run",
            summary=f"Real build+eval of {exp[:12]} ({hyp}) — large compute",
            detail={"lm": spec.lm, "vision": spec.vision,
                    "align_steps": align_steps, "distill_steps": distill_steps},
            by="construction_loop")
        print(f"  ⏸ awaiting operator approval (id {aid}) — approve in the console's "
              f"Approvals tab, or: python -m services.approvals approve {aid}")
        decision = approvals.wait_for_approval(aid)
        if decision != "approved":
            print(f"  ✗ run {decision} by operator — aborting (nothing built).")
            return {"experiment_id": exp, "status": f"approval_{decision}",
                    "hypothesis_id": (queue_entry or {}).get("hypothesis_id", "P2-B1")}
        print(f"  ✓ approved — proceeding with the build.")

    from runners.build_student import build  # lazy: pulls torch/transformers

    out_dir = out_dir or (PROJECT_ROOT / "artifacts" / "students" /
                          f"build_{spec.content_hash()[:12]}")

    from services.runlog import tee_stdout
    with tee_stdout(f"construction_{spec.content_hash()[:12]}"):  # standard run log
        record = build(spec, out_dir, smoke=smoke, align_steps=align_steps,
                       distill_steps=distill_steps, max_samples=max_samples)

        # B1.3: score the constructed student on the same path as the LFM2 benchmark.
        if eval_after and not smoke and record.get("student_dir"):
            from runners.eval_student import evaluate
            eval_out = out_dir / "eval"
            results = evaluate(out_dir, spec.eval.benchmarks, eval_n or spec.eval.n, eval_out)
            record["quality_scores"] = [
                {"benchmark": b, "metric": "Overall", "value": r["scores"].get("Overall"),
                 "delta_vs_benchmark": r["delta_vs_benchmark"]}
                for b, r in results.items()
            ]

    entry = _ledger_entry(spec, record, queue_entry)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = LEDGER_DIR / f"construction_{spec.content_hash()[:12]}.json"
    ledger_path.write_text(json.dumps(entry, indent=2))
    try:
        shown = ledger_path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = ledger_path
    print(f"  ✅ recorded → {shown}")
    print(f"     status={entry['report']['status']}  "
          f"hypothesis={entry['hypothesis_id']}  exp={spec.content_hash()[:12]}")
    print("  ↻ the Search Strategist's next propose_next() will read this ledger entry "
          "and re-route.")
    return entry


def main():
    ap = argparse.ArgumentParser(description="Run the next queued student construction (ADR-0012 B1.2)")
    ap.add_argument("--spec", default=None, help="Explicit StudentSpec JSON (overrides the queue)")
    ap.add_argument("--smoke", action="store_true", help="Tiny end-to-end build (proves the loop)")
    ap.add_argument("--eval", action="store_true", help="Run same-path MCQ eval after a real build (B1.3)")
    ap.add_argument("--eval-n", type=int, default=None, help="Override eval samples per benchmark")
    ap.add_argument("--align-steps", type=int, default=None, help="Cap align steps (real run budget)")
    ap.add_argument("--distill-steps", type=int, default=None, help="Cap distill steps (real run budget)")
    ap.add_argument("--max-samples", type=int, default=None, help="Cap train rows")
    ap.add_argument("--require-approval", action="store_true",
                    help="Gate a real run behind the operator approval queue (blocks until approved)")
    args = ap.parse_args()

    if args.spec:
        spec = StudentSpec.model_validate_json(Path(args.spec).read_text())
        entry = None
    else:
        spec, entry = next_proposed_spec()
        src = "agent queue" if entry else "default P2-B1 spec (queue empty)"
        print(f"▶ construction_loop  source={src}  spec={spec.content_hash()[:12]}")

    run_once(spec, smoke=args.smoke, queue_entry=entry,
             eval_after=args.eval, eval_n=args.eval_n,
             align_steps=args.align_steps, distill_steps=args.distill_steps,
             max_samples=args.max_samples, require_approval=args.require_approval)


if __name__ == "__main__":
    main()
