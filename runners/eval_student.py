"""
runners/eval_student.py
───────────────────────
Phase 2 / ADR-0012 B1.3 — score a CONSTRUCTED student on the same MCQ path as the
LFM2-VL-450M benchmark (P2-1.3 methodology). Reuses eval_vlmeval's dataset loading,
prompting, and VLMEvalKit scoring — the assembled StudentVLM just implements the
same `infer(image_path, question, is_mcq)` protocol as the registry models, so the
inference path is held constant for a valid comparison.

Usage
-----
    python runners/eval_student.py --build artifacts/students/build_<hash> \
        --benchmarks POPE RealWorldQA MMBench_DEV_EN --n 100 \
        --out results/phase2_b1_3_eval
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# LFM2-VL-450M benchmark on the same fp16 path (P2-D2 same-session numbers).
BENCHMARK_BAR = {"POPE": 87.7, "RealWorldQA": 0.42, "MMBench_DEV_EN": 0.74}


def _chance_floor(ds, df, bench: str, scratch: Path, is_mcq: bool) -> dict:
    """Strongest trivial constant-prediction baseline, via the SAME scorer.

    A model at or below this floor has demonstrated no real ability — a raw score
    must be read against it (always-'A' on a 41%-A slice already scores 0.41).
    """
    from runners.eval_vlmeval import score_benchmark
    best = {"label": None, "Overall": None}
    for c in (["A", "B", "C", "D"] if is_mcq else ["Yes", "No"]):
        rdf = df.copy()
        rdf["prediction"] = c
        rdf = rdf.drop(columns=["image"], errors="ignore")
        try:
            sc = score_benchmark(ds, rdf, scratch, f"floor_{bench}_{c}")
        except Exception:
            continue
        ov = sc.get("Overall", sc.get("acc"))
        if ov is not None and (best["Overall"] is None or ov > best["Overall"]):
            best = {"label": f"always-{c}", "Overall": ov}
    return best


def evaluate(build_dir: Path, benchmarks: list[str], n: int, out_dir: Path) -> dict:
    """Score the constructed student; write MetricsReport JSONs; return {bench: scores}.

    Each score is reported AGAINST its chance floor (and, for POPE, as balanced
    accuracy — the trustworthy metric on imbalanced slices, vs the F1 'Overall')."""
    import json
    from runners.build_student import load_student
    from runners.eval_vlmeval import (
        get_dataset_slice, get_image_path, build_mcq_question, score_benchmark,
    )
    from services import run_control as rc

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_eval_scratch"; scratch.mkdir(exist_ok=True)

    print(f"  loading constructed student from {build_dir} …")
    model = load_student(build_dir)

    results: dict[str, dict] = {}
    for bench in benchmarks:
        print(f"\n  Benchmark: {bench} ({n} samples)")
        ds, df = get_dataset_slice(bench, n)
        preds, failed = [], 0
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 10 == 0:
                print(f"    [{i}/{n}] …", end="\r", flush=True)
            try:
                rc.check()  # operator stop/kill honored between samples
            except rc.RunStopped as e:
                print(f"\n    operator '{e.mode}' during eval — halting")
                raise
            try:
                question, is_mcq = build_mcq_question(row)
                preds.append(model.infer(get_image_path(ds, row, bench), question, is_mcq))
            except Exception as exc:
                print(f"\n    WARN row {row.get('index', i)}: {exc}")
                preds.append(""); failed += 1
        result_df = df.copy()
        result_df["prediction"] = preds
        result_df = result_df.drop(columns=["image"], errors="ignore")
        scores = score_benchmark(ds, result_df, scratch, f"student_{bench}")
        bar = BENCHMARK_BAR.get(bench)
        overall = scores.get("Overall")
        delta = (overall - bar) if (bar is not None and overall is not None) else None

        is_mcq = build_mcq_question(df.iloc[0])[1]
        floor = _chance_floor(ds, df, bench, scratch, is_mcq)
        bal_acc = None if is_mcq else scores.get("acc")  # POPE: balanced acc

        # Judge on the TRUSTWORTHY metric per benchmark: balanced accuracy (floor 50)
        # for POPE — its F1 "Overall" is misleading on imbalanced slices; accuracy vs
        # the majority floor for MCQ.
        if is_mcq:
            metric, mval, mfloor = "accuracy", overall, floor["Overall"]
        else:
            metric, mval, mfloor = "balanced-accuracy", bal_acc, 50.0
        above_floor = (mval is not None and mfloor is not None and mval > mfloor + 1e-9)

        print(f"    Overall(raw)={overall}  | trustworthy {metric}={mval}  "
              f"vs floor {mfloor}  → above_floor={above_floor}")
        if not above_floor:
            print(f"    ⚠️ AT/BELOW the chance floor — no demonstrated ability on {bench}.")

        rec = {"benchmark": bench, "scores": scores, "benchmark_bar": bar,
               "delta_vs_benchmark": delta, "chance_floor": floor,
               "trustworthy_metric": metric, "trustworthy_value": mval,
               "trustworthy_floor": mfloor, "above_floor": above_floor,
               "balanced_accuracy": bal_acc, "failed": failed,
               "evaluated_at": datetime.now(timezone.utc).isoformat()}
        (out_dir / f"student_{bench}.json").write_text(json.dumps(rec, indent=2))
        results[bench] = rec

    print(f"\n  ✅ student eval done → {out_dir}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Same-path MCQ eval for a constructed student (B1.3)")
    ap.add_argument("--build", required=True, help="Build dir (contains student/)")
    ap.add_argument("--benchmarks", nargs="+", default=["POPE", "RealWorldQA", "MMBench_DEV_EN"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/phase2_b1_3_eval")
    args = ap.parse_args()
    evaluate(Path(args.build), args.benchmarks, args.n, Path(args.out))


if __name__ == "__main__":
    main()
