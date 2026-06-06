"""
services/pareto_tracker.py
──────────────────────────
Pareto Tracker for Phase 1 VLM optimization.

Reads all MetricsReport records from the experiment ledger, computes
a Pareto frontier across configurable axes, and writes:
  - artifacts/pareto/pareto_frontier.json  ← machine-readable frontier
  - artifacts/pareto/pareto_history.json   ← all experiments with domination flags

Primary axes (defaults):
  - CLIP-score  (higher is better — quality)
  - TTFT ms     (lower is better — latency)
  - TPS         (higher is better — throughput)
  - Peak mem MB (lower is better — memory)

A point P dominates Q if P is at least as good on ALL axes and strictly
better on at least one.  The Pareto frontier is the set of non-dominated
points.

Usage:
    from services.pareto_tracker import ParetoTracker

    tracker = ParetoTracker()
    frontier = tracker.update()   # reads ledger, writes JSON, returns frontier
    tracker.print_report()        # console summary
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Project root ──────────────────────────────────────────────────────────────

_HERE        = Path(__file__).parent
PROJECT_ROOT = _HERE.parent
LEDGER_DIR   = PROJECT_ROOT / "artifacts" / "experiment_ledger"
PARETO_DIR   = PROJECT_ROOT / "artifacts" / "pareto"

# ── Phase 0 baselines (anchors — not in ledger) ───────────────────────────────

PHASE0_BASELINES = [
    {
        "experiment_id":        "ph0_lfm2_q4_0",
        "label":                "Ph0 LFM2-Q4_0",
        "model_key":            "LFM2-VL-450M",
        "quantization":         "Q4_0",
        "clip_score":           27.60,
        "ttft_ms":              None,   # anomalous pre-harness-fix; excluded from TTFT axis
        "decode_tokens_per_sec": 82.4,
        "peak_memory_mb":       275.0,
        "on_disk_size_mb":      279.0,
        "is_baseline":          True,
    },
    {
        "experiment_id":        "ph0_smolvlm_q4km",
        "label":                "Ph0 SmolVLM-Q4KM",
        "model_key":            "SmolVLM-500M",
        "quantization":         "Q4_K_M",
        "clip_score":           24.11,
        "ttft_ms":              20.15,
        "decode_tokens_per_sec": 48.6,
        "peak_memory_mb":       367.0,
        "on_disk_size_mb":      393.0,
        "is_baseline":          True,
    },
]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ExperimentPoint:
    """One experiment result expressed as a multi-dimensional point."""
    experiment_id:          str
    label:                  str
    model_key:              str
    quantization:           str
    clip_score:             Optional[float]
    ttft_ms:                Optional[float]
    decode_tokens_per_sec:  Optional[float]
    peak_memory_mb:         Optional[float]
    on_disk_size_mb:        Optional[float]
    is_baseline:            bool = False
    # Set by tracker
    is_pareto_frontier:     bool = False
    dominated_by:           list[str] = field(default_factory=list)

    def dominates(self, other: "ExperimentPoint", axes: list[str]) -> bool:
        """
        Return True if self Pareto-dominates other on the given axes.

        Axes: "clip_score" (higher better), "ttft_ms" (lower better),
              "decode_tokens_per_sec" (higher better), "peak_memory_mb" (lower better).

        Dominance rule: self dominates other only if, for EVERY axis where
        `other` has a non-null value, self is at least as good (and strictly
        better on at least one such shared axis). If self is missing a value
        on an axis where other has data, self cannot dominate other — missing
        data is treated as "unknown, not better". This prevents Mac-only proxy
        experiments (no TTFT/Mem) from dominating iPhone-measured points.
        """
        HIGHER_BETTER = {"clip_score", "decode_tokens_per_sec"}
        LOWER_BETTER  = {"ttft_ms", "peak_memory_mb", "on_disk_size_mb"}

        at_least_as_good = True
        strictly_better  = False
        n_compared = 0

        for ax in axes:
            v_self  = getattr(self, ax)
            v_other = getattr(other, ax)

            if v_other is None:
                continue   # other has no data on this axis — skip

            if v_self is None:
                # other has data here but self doesn't — self cannot dominate
                at_least_as_good = False
                break

            n_compared += 1
            if ax in HIGHER_BETTER:
                if v_self < v_other - 1e-9:
                    at_least_as_good = False
                    break
                if v_self > v_other + 1e-9:
                    strictly_better = True
            elif ax in LOWER_BETTER:
                if v_self > v_other + 1e-9:
                    at_least_as_good = False
                    break
                if v_self < v_other - 1e-9:
                    strictly_better = True

        # Need at least one axis compared; no data in common → cannot dominate
        return n_compared > 0 and at_least_as_good and strictly_better


# ── Ledger reader ─────────────────────────────────────────────────────────────

def _load_ledger_points() -> list[ExperimentPoint]:
    """Parse all JSON files in the experiment ledger into ExperimentPoints."""
    points: list[ExperimentPoint] = []

    for path in sorted(LEDGER_DIR.glob("*.json")):
        if path.stem.endswith("_preds"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        report = data.get("report", {})
        config = data.get("config", {})

        if report.get("status") not in ("completed",):
            continue

        # Extract quality scores
        qs = {s["metric"]: s["value"] for s in report.get("quality_scores", [])}
        clip = qs.get("clip_score_mean")

        # Model key from notes or model_id
        model_id = config.get("model_id", "")
        if "LFM2" in model_id or "lfm2" in model_id.lower():
            model_key = "LFM2-VL-450M"
        elif "SmolVLM" in model_id or "smolvlm" in model_id.lower():
            model_key = "SmolVLM-500M"
        elif "MiniCPM" in model_id:
            model_key = "MiniCPM-V-4.6"
        else:
            model_key = model_id.split("/")[-1]

        quant = config.get("compression", {}).get("weight_dtype", "?")
        res   = config.get("input_resolution")
        label_parts = [model_key, quant]
        if res:
            label_parts.append(f"{res}px")
        label = " · ".join(label_parts)

        points.append(ExperimentPoint(
            experiment_id=report["experiment_id"],
            label=label,
            model_key=model_key,
            quantization=quant,
            clip_score=clip,
            ttft_ms=report.get("ttft_ms"),
            decode_tokens_per_sec=report.get("decode_tokens_per_sec"),
            peak_memory_mb=report.get("peak_memory_mb"),
            on_disk_size_mb=report.get("on_disk_size_mb"),
        ))

    return points


# ── Pareto computation ────────────────────────────────────────────────────────

DEFAULT_AXES = ["clip_score", "ttft_ms", "decode_tokens_per_sec", "peak_memory_mb"]


def _compute_pareto(
    points: list[ExperimentPoint],
    axes: list[str] = DEFAULT_AXES,
) -> list[ExperimentPoint]:
    """
    Mark each point as Pareto-frontier or dominated.
    Returns the same list with is_pareto_frontier / dominated_by set.
    """
    for p in points:
        p.is_pareto_frontier = True
        p.dominated_by = []

    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            if q.dominates(p, axes):
                p.is_pareto_frontier = False
                p.dominated_by.append(q.experiment_id)

    return points


# ── ParetoTracker ─────────────────────────────────────────────────────────────

class ParetoTracker:
    """
    Loads the experiment ledger, computes the Pareto frontier, and
    writes summary JSON files.

    Args:
        axes:  axes to use for Pareto dominance check.
               Default: clip_score, ttft_ms, decode_tokens_per_sec, peak_memory_mb.
    """

    def __init__(self, axes: list[str] = DEFAULT_AXES):
        self.axes   = axes
        self.points: list[ExperimentPoint] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self) -> list[ExperimentPoint]:
        """
        Read ledger → compute frontier → write JSON files → return frontier points.
        """
        # Load Phase 0 baselines + ledger experiments
        baseline_points = [ExperimentPoint(**b) for b in PHASE0_BASELINES]
        ledger_points   = _load_ledger_points()
        all_points      = baseline_points + ledger_points

        # Compute Pareto frontier
        all_points = _compute_pareto(all_points, self.axes)

        self.points = all_points

        # Persist
        PARETO_DIR.mkdir(parents=True, exist_ok=True)
        self._write_frontier_json()
        self._write_history_json()

        frontier = [p for p in all_points if p.is_pareto_frontier]
        return frontier

    def print_report(self) -> None:
        """Print a formatted Pareto frontier report to stdout."""
        if not self.points:
            print("  (no data — call update() first)")
            return

        frontier  = [p for p in self.points if p.is_pareto_frontier]
        dominated = [p for p in self.points if not p.is_pareto_frontier]

        print("\n" + "═" * 78)
        print("  Pareto Frontier")
        print("═" * 78)
        print(f"  {'Label':<38} {'CLIP':>6} {'TTFT ms':>9} {'TPS':>7} {'Mem MB':>8}")
        print("  " + "─" * 72)
        for p in sorted(frontier, key=lambda x: (x.clip_score or 0), reverse=True):
            cs = f"{p.clip_score:.2f}" if p.clip_score else "—"
            ts = f"{p.ttft_ms:.1f}" if p.ttft_ms else "—"
            ps = f"{p.decode_tokens_per_sec:.1f}" if p.decode_tokens_per_sec else "—"
            ms = f"{p.peak_memory_mb:.0f}" if p.peak_memory_mb else "—"
            tag = " ⭐" if not p.is_baseline else ""
            print(f"  {(p.label + tag):<38} {cs:>6} {ts:>9} {ps:>7} {ms:>8}")

        print()
        print(f"  Dominated ({len(dominated)}):")
        for p in sorted(dominated, key=lambda x: (x.clip_score or 0), reverse=True):
            cs = f"{p.clip_score:.2f}" if p.clip_score else "—"
            ts = f"{p.ttft_ms:.1f}" if p.ttft_ms else "—"
            dom_by = ", ".join(x[:8] + "…" for x in p.dominated_by[:2])
            print(f"    {p.label:<36} CLIP={cs}  TTFT={ts}  (dominated by {dom_by})")

        print()
        print(f"  Phase 1 exit criterion 1.4: ", end="")
        ph0_clip = max(
            (p.clip_score or 0) for p in self.points
            if p.is_baseline and p.model_key == "LFM2-VL-450M"
        )
        best_non_baseline = max(
            (p.clip_score or 0) for p in self.points
            if not p.is_baseline and p.model_key == "LFM2-VL-450M" and p.clip_score
        )
        if best_non_baseline > ph0_clip:
            print(f"MET ✅  (best LFM2 CLIP {best_non_baseline:.2f} > baseline {ph0_clip:.2f})")
        else:
            print(f"NOT MET ❌  (best LFM2 CLIP {best_non_baseline:.2f} ≤ baseline {ph0_clip:.2f})")
        print("═" * 78)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _write_frontier_json(self) -> Path:
        frontier = [p for p in self.points if p.is_pareto_frontier]
        payload  = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "axes":         self.axes,
            "n_total":      len(self.points),
            "n_frontier":   len(frontier),
            "frontier": [
                {
                    "experiment_id":          p.experiment_id,
                    "label":                  p.label,
                    "model_key":              p.model_key,
                    "quantization":           p.quantization,
                    "clip_score":             p.clip_score,
                    "ttft_ms":                p.ttft_ms,
                    "decode_tokens_per_sec":  p.decode_tokens_per_sec,
                    "peak_memory_mb":         p.peak_memory_mb,
                    "on_disk_size_mb":        p.on_disk_size_mb,
                    "is_baseline":            p.is_baseline,
                }
                for p in sorted(frontier, key=lambda x: (x.clip_score or 0), reverse=True)
            ],
        }
        path = PARETO_DIR / "pareto_frontier.json"
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _write_history_json(self) -> Path:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "axes":         self.axes,
            "experiments": [
                {
                    "experiment_id":          p.experiment_id,
                    "label":                  p.label,
                    "model_key":              p.model_key,
                    "quantization":           p.quantization,
                    "clip_score":             p.clip_score,
                    "ttft_ms":                p.ttft_ms,
                    "decode_tokens_per_sec":  p.decode_tokens_per_sec,
                    "peak_memory_mb":         p.peak_memory_mb,
                    "on_disk_size_mb":        p.on_disk_size_mb,
                    "is_baseline":            p.is_baseline,
                    "is_pareto_frontier":     p.is_pareto_frontier,
                    "dominated_by":           p.dominated_by,
                }
                for p in self.points
            ],
        }
        path = PARETO_DIR / "pareto_history.json"
        path.write_text(json.dumps(payload, indent=2))
        return path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    tracker = ParetoTracker()
    frontier = tracker.update()
    tracker.print_report()
    print(f"\n  Wrote: {PARETO_DIR}/pareto_frontier.json")
    print(f"  Wrote: {PARETO_DIR}/pareto_history.json")
