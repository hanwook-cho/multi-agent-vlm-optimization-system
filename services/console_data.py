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
HYPOTHESIS_QUEUE = PROJECT_ROOT / "artifacts" / "hypothesis_queue.json"  # Mode B (ADR-0015)
APPROVAL_LOG = PROJECT_ROOT / "artifacts" / "approval_log.json"  # H3 — may not exist
RUN_YAML = PROJECT_ROOT / "run.yaml"
RUN_LOG_DIR = PROJECT_ROOT / "artifacts" / "logs"          # standard run logs (services.runlog)
_LEGACY_LOG = Path("/tmp/b13_build.log")                    # earlier ad-hoc default


def default_log_path() -> str:
    """Newest run log in the standard dir; fall back to the legacy /tmp path or ''.

    So the console points at the current run automatically (runs tee here via
    services.runlog) without the operator wiring a path.
    """
    if RUN_LOG_DIR.exists():
        logs = sorted(RUN_LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            return str(logs[0])
    return str(_LEGACY_LOG) if _LEGACY_LOG.exists() else ""

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


def log_tail(path: Path, n: int = 24, contains: str | None = None) -> str:
    """Last n lines of a log file (carriage-returns flattened for tqdm output).
    `contains` filters to lines containing that substring (case-insensitive) — so the
    log view can hide unrelated lines (e.g. show only 'verified', 'distill', 'error')."""
    if not path or not Path(path).exists():
        return ""
    raw = Path(path).read_text(errors="replace").replace("\r", "\n")
    lines = [l for l in raw.splitlines() if l.strip()]
    if contains:
        c = contains.lower()
        lines = [l for l in lines if c in l.lower()]
    return "\n".join(lines[-n:])


def recent_logs(n: int = 12) -> list[tuple[str, str]]:
    """Recent run logs → [(label, path)] newest first. Label is '<name> · <age> ago'
    so the operator can pick a specific run (construction_… / research_… / eval_…)."""
    if not RUN_LOG_DIR.exists():
        return []
    import time
    now = time.time()
    out = []
    for p in sorted(RUN_LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:n]:
        age = now - p.stat().st_mtime
        a = f"{int(age)}s" if age < 60 else (f"{int(age // 60)}m" if age < 3600 else f"{int(age // 3600)}h")
        out.append((f"{p.stem} · {a} ago", str(p)))
    return out


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


def local_server_up(host: str = "localhost", port: int = 8080, timeout: float = 0.3) -> bool:
    """True if the local strategist (llama.cpp) server is reachable on host:port."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def hypothesis_rows() -> list[dict]:
    """The Search Strategist's hypothesis table, compact, for the rationale view (H4)."""
    try:
        from agents.search_strategist import HYPOTHESIS_TABLE
    except Exception:
        return []
    return [{"id": h.get("id"), "status": h.get("status"), "phase": h.get("phase"),
             "technique": h.get("technique")} for h in HYPOTHESIS_TABLE]


def latest_proposals(n: int = 6) -> list[dict]:
    """Recent agent proposals (with rationale) from the experiment + construction queues."""
    out: list[dict] = []
    for q, kind in ((CONSTRUCTION_QUEUE, "construction"), (EXPERIMENT_QUEUE, "experiment")):
        if not q.exists():
            continue
        try:
            items = json.loads(q.read_text())
        except Exception:
            items = []
        for it in items if isinstance(items, list) else []:
            out.append({
                "hypothesis": it.get("hypothesis_id", ""),
                "rationale": it.get("rationale", ""),
                "proposed_at": it.get("proposed_at", ""),
                "kind": kind,
            })
    out.sort(key=lambda r: r["proposed_at"], reverse=True)
    return out[:n]


def pending_approvals(path: Path = APPROVAL_LOG) -> list[dict]:
    """Pending items from the approval log (H3). Empty until that log exists."""
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text())
        return [a for a in items if a.get("status", "pending") == "pending"]
    except Exception:
        return []


# ── Launch (the one action, not a read) ──────────────────────────────────────
# The console is otherwise a pure reader; this lets the operator START the next
# queued construction from the UI. It shells out to the SAME entry point used on
# the CLI (services/construction_loop.py) — no separate code path — so a run
# launched here is identical to one launched by hand.

def build_construction_cmd(*, smoke: bool = False, eval_after: bool = True,
                           seed: int = 0, require_approval: bool = False,
                           python: str | None = None) -> list[str]:
    """Pure: assemble the construction_loop command (kept separate for testing)."""
    import sys
    py = python or sys.executable
    cmd = [py, str(PROJECT_ROOT / "services" / "construction_loop.py")]
    if smoke:
        cmd.append("--smoke")
    if eval_after:
        cmd.append("--eval")
    if require_approval:
        cmd.append("--require-approval")
    cmd += ["--seed", str(int(seed))]
    return cmd


def recent_hypotheses(n: int = 6) -> list[dict]:
    """Verified Mode-B hypothesis records (artifacts/hypothesis_queue.json) → compact rows."""
    if not HYPOTHESIS_QUEUE.exists():
        return []
    try:
        items = json.loads(HYPOTHESIS_QUEUE.read_text())
    except Exception:
        return []
    rows = []
    for it in (items if isinstance(items, list) else [])[-n:][::-1]:
        r = it.get("record", {}) or {}
        sc = r.get("source_citation", {}) or {}
        ac = r.get("applicability_check", {}) or {}
        rows.append({
            "technique": it.get("title") or r.get("title", ""),
            "arxiv": sc.get("arxiv_id", ""),
            "applies": ac.get("verdict", ""),
            "claimed_effect": (r.get("claimed_effect", "") or "")[:110],
            "found_at": (it.get("proposed_at", "") or "")[:19],
        })
    return rows


def build_research_cmd(*, query: str, problem: str | None = None, max_papers: int = 5,
                       backend: str = "auto", python: str | None = None) -> list[str]:
    """Pure: assemble the research_analyst command (kept separate for testing)."""
    import sys
    py = python or sys.executable
    cmd = [py, str(PROJECT_ROOT / "agents" / "research_analyst.py"),
           "--query", query, "--max-papers", str(int(max_papers)), "--backend", backend]
    if problem:
        cmd += ["--problem", problem]
    return cmd


def launch_research(*, query: str, problem: str | None = None, max_papers: int = 5,
                    backend: str = "auto") -> dict:
    """Spawn the Research Analyst (Mode B) as a detached subprocess. Survivors land in
    the hypothesis queue (shown by recent_hypotheses) + the escalation gate."""
    import subprocess
    cmd = build_research_cmd(query=query, problem=problem, max_papers=max_papers, backend=backend)
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return {"pid": proc.pid, "cmd": " ".join(cmd)}


def launch_construction(*, smoke: bool = False, eval_after: bool = True,
                        seed: int = 0, require_approval: bool = False) -> dict:
    """Spawn construction_loop as a detached subprocess (survives Streamlit reruns).

    The loop tees its output to artifacts/logs/ via services.runlog, so the Monitor
    (which auto-points at the newest log) shows progress without extra wiring.
    Returns {pid, cmd}. It builds the most recent queued spec (or the default if the
    queue is empty), exactly as the CLI does.
    """
    import subprocess
    cmd = build_construction_cmd(smoke=smoke, eval_after=eval_after,
                                 seed=seed, require_approval=require_approval)
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            start_new_session=True)   # detach from the Streamlit process
    return {"pid": proc.pid, "cmd": " ".join(cmd)}
