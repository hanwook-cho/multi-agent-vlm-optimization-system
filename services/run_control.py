"""
services/run_control.py
───────────────────────
Operator run controls (ADR-0013 H1) — pause / stop / kill for long-running loops.

The HLD's resumable job-queue model (§7.4) means a halted run costs at most one
step. This implements that: long loops call `checkpoint()` between steps; the
operator sets the control state from the CLI (or the future GUI button), and the
loop reacts at the next checkpoint — never mid-step, never by a blind process kill.

States:
  run    — proceed (default; also "resume")
  pause  — block at the next checkpoint until the operator resumes
  stop   — GRACEFUL halt: finish nothing more, exit the loop, let the caller save
  kill   — IMMEDIATE abort: raise past the caller; nothing is saved

CLI:
    python -m services.run_control status
    python -m services.run_control pause   "lunch"
    python -m services.run_control resume
    python -m services.run_control stop     "good enough"
    python -m services.run_control kill      "swap-thrashing"
    python -m services.run_control clear     # reset to run

In code:
    from services import run_control as rc
    for step in ...:
        rc.checkpoint()          # blocks if paused; raises RunStopped on stop/kill
        ... do one step ...
  caught by the loop:
    except rc.RunStopped as e:
        if e.mode == "kill": raise         # abort, no save
        break                              # stop: graceful, caller saves
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONTROL_FILE = PROJECT_ROOT / "artifacts" / "run_control.json"

_VALID = {"run", "pause", "stop", "kill"}


class RunStopped(Exception):
    """Raised by checkpoint() when the operator requested stop or kill."""
    def __init__(self, mode: str, reason: str = ""):
        self.mode = mode
        self.reason = reason
        super().__init__(f"run {mode}" + (f": {reason}" if reason else ""))


def set_state(state: str, reason: str = "") -> dict:
    if state == "resume":
        state = "run"
    if state not in _VALID:
        raise ValueError(f"invalid state '{state}' — one of {sorted(_VALID)} (or 'resume')")
    CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "reason": reason,
               "ts": datetime.now(timezone.utc).isoformat()}
    CONTROL_FILE.write_text(json.dumps(payload, indent=2))
    return payload


def get_state() -> dict:
    if not CONTROL_FILE.exists():
        return {"state": "run", "reason": "", "ts": None}
    try:
        d = json.loads(CONTROL_FILE.read_text())
        if d.get("state") in _VALID:
            return d
    except Exception:
        pass
    return {"state": "run", "reason": "", "ts": None}


def clear() -> None:
    CONTROL_FILE.unlink(missing_ok=True)


def check() -> None:
    """Raise RunStopped if the operator requested stop or kill. Non-blocking."""
    d = get_state()
    if d["state"] in ("stop", "kill"):
        raise RunStopped(d["state"], d.get("reason", ""))


def wait_if_paused(poll: float = 2.0, sleep=time.sleep) -> None:
    """Block while state == pause. A stop/kill during pause is honored on resume."""
    while get_state()["state"] == "pause":
        sleep(poll)
    check()


def checkpoint(poll: float = 2.0, sleep=time.sleep) -> None:
    """Call between steps: blocks if paused, raises RunStopped on stop/kill."""
    wait_if_paused(poll=poll, sleep=sleep)


def main():
    ap = argparse.ArgumentParser(description="Operator run controls (ADR-0013 H1)")
    ap.add_argument("action", choices=sorted(_VALID) + ["resume", "status", "clear"])
    ap.add_argument("reason", nargs="?", default="", help="Optional note recorded with the state")
    args = ap.parse_args()

    if args.action == "status":
        print(json.dumps(get_state(), indent=2))
    elif args.action == "clear":
        clear()
        print("run control cleared → run")
    else:
        p = set_state(args.action, args.reason)
        print(f"run control set → {p['state']}" + (f"  ({p['reason']})" if p['reason'] else ""))


if __name__ == "__main__":
    main()
