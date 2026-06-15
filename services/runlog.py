"""
services/runlog.py
──────────────────
Standard run-log convention (ADR-0013 H2 follow-up). Long runs tee their stdout to
artifacts/logs/<name>.log so the operator console can find and tail the current run
without the operator manually wiring a log path.

    from services.runlog import tee_stdout, RUN_LOG_DIR
    with tee_stdout("construction_df64c49b"):
        ...   # everything printed also lands in artifacts/logs/construction_df64c49b.log
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RUN_LOG_DIR = PROJECT_ROOT / "artifacts" / "logs"


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def log_path_for(name: str) -> Path:
    return RUN_LOG_DIR / f"{name}.log"


@contextmanager
def tee_stdout(name: str):
    """Duplicate stdout to artifacts/logs/<name>.log for the duration of the block."""
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = log_path_for(name)
    f = open(path, "a", buffering=1)
    old = sys.stdout
    sys.stdout = _Tee(old, f)
    try:
        print(f"# run log → {path}")
        yield path
    finally:
        sys.stdout = old
        f.close()
