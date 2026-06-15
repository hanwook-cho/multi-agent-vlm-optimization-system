"""
services/console_data.py
────────────────────────
Pure data helpers for the operator console (ADR-0013 H2). No Streamlit here, so
these are unit-testable; `operator_console.py` is the thin view that renders them.

Everything reads the shared state the rest of the system already writes: the run
control flag, the experiment/construction queues, the experiment ledger, run logs,
and run.yaml. Single source of truth — the console is just a reader.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LEDGER_DIR = PROJECT_ROOT / "artifacts" / "experiment_ledger"
CONSTRUCTION_QUEUE = PROJECT_ROOT / "artifacts" / "construction_queue.json"
EXPERIMENT_QUEUE = PROJECT_ROOT / "artifacts" / "experiment_queue.json"
APPROVAL_LOG = PROJECT_ROOT / "artifacts" / "approval_log.json"  # H3 — may not exist
RUN_YAML = PROJECT_ROOT / "run.yaml"

_PROGRESS_RE = re.compile(r"\[(align|distill)\]\s+step\s+(\d+)/(\d+).*?loss=([0-9.]+)")


def queue_len(path: Path) -> int:
    """Number of items in a JSON-array queue file (0 if missing/unreadable)."""
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def backend_label(run_yaml: Path | None = None, env: dict | None = None) -> str:
    """Resolve the agent/chat backend to 'local' or 'api' (default local, ADR-0013).

    Opt-in to API only via STRATEGIST_BACKEND env or run.yaml chat_backend.
    """
    env = env if env is not None else {}
    choice = (env.get("STRATEGIST_BACKEND") or "").strip().lower()
    if not choice and run_yaml is not None and run_yaml.exists():
        try:
            import yaml
            choice = (yaml.safe_load(run_yaml.read_text()) or {}).get("chat_backend", "")
        except Exception:
            choice = ""
    choice = (choice or "local").lower()
    return "api" if choice in ("api", "anthropic") else "local"


def parse_progress(text: str) -> dict | None:
    """Latest training progress from a run log: {stage, step, total, loss} or None."""
    matches = _PROGRESS_RE.findall(text or "")
    if not matches:
        return None
    stage, step, total, loss = matches[-1]
    return {"stage": stage, "step": int(step), "total": int(total), "loss": float(loss)}


def log_tail(path: Path, n: int = 24) -> str:
    """Last n lines of a log file (carriage-returns flattened for tqdm output)."""
    if not path or not Path(path).exists():
        return ""
    raw = Path(path).read_text(errors="replace").replace("\r", "\n")
    lines = [l for l in raw.splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def recent_constructions(ledger_dir: Path = LEDGER_DIR, n: int = 8) -> list[dict]:
    """Recent student_construction ledger entries → compact rows for the table."""
    rows: list[dict] = []
    if not ledger_dir.exists():
        return rows
    for path in ledger_dir.glob("construction_*.json"):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        rep = d.get("report", {})
        scores = {s.get("benchmark"): s.get("value")
                  for s in rep.get("quality_scores", []) if s.get("metric") == "Overall"}
        rows.append({
            "experiment_id": (d.get("experiment_id") or "")[:12],
            "hypothesis": d.get("hypothesis_id", ""),
            "status": rep.get("status", ""),
            "recorded_at": d.get("recorded_at", ""),
            "POPE": scores.get("POPE"),
            "RealWorldQA": scores.get("RealWorldQA"),
            "MMBench_DEV_EN": scores.get("MMBench_DEV_EN"),
        })
    rows.sort(key=lambda r: r["recorded_at"], reverse=True)
    return rows[:n]


def pending_approvals(path: Path = APPROVAL_LOG) -> list[dict]:
    """Pending items from the approval log (H3). Empty until that log exists."""
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text())
        return [a for a in items if a.get("status", "pending") == "pending"]
    except Exception:
        return []
