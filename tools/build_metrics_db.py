"""
build_metrics_db.py
───────────────────
Scan all Phase 0 artifact JSON files and load them into a SQLite database
at metrics.db.  Safe to re-run — it clears and rebuilds every time.

Tables:
  iphone_perf   — iPhone 16 Pro TTFT / TPS / memory results (Task 3.x)
  mac_quality   — Mac benchmark accuracy results (Task 2.2)
  clip_scores   — CLIP-score results (runners/compute_clip_score.py)

Usage:
  python tools/build_metrics_db.py [--db metrics.db]
"""

import argparse
import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_iphone_perf(db: sqlite3.Connection) -> int:
    """Load iPhone MetricsReport JSONs from artifacts/eval_task_3_*/"""
    db.execute("""
        CREATE TABLE IF NOT EXISTS iphone_perf (
            model_key         TEXT NOT NULL,
            device_id         TEXT NOT NULL,
            quantization      TEXT,
            runtime           TEXT,
            ttft_ms_mean      REAL,
            ttft_ms_stddev    REAL,
            tps_mean          REAL,
            peak_memory_mb    REAL,
            on_disk_mb        REAL,
            mmproj_mb         REAL,
            timestamp         TEXT,
            experiment_id     TEXT,
            notes             TEXT
        )
    """)
    db.execute("DELETE FROM iphone_perf")

    rows = 0
    for path in sorted(PROJECT_ROOT.glob("artifacts/eval_task_3_*/*.json")):
        data = json.loads(path.read_text())
        perf = data.get("performance_metrics", {})

        # Normalise camelCase (FastVLM/MLX harness) vs snake_case (VLMHarness)
        def g(snake, camel, src=data):
            return src.get(snake) or src.get(camel)

        model_key = g("model_key", "modelKey")
        device_id = g("device_id", "device") or "iphone_16_pro"
        quant     = g("quantization", "quantization")
        runtime   = g("runtime", "runtime") or g("runtime", "backend")
        timestamp = g("timestamp", "timestamp")
        exp_id    = g("experiment_id", "experimentId")
        notes     = g("notes", "notes") or ""

        # Performance fields — check perf sub-dict first, then top-level camelCase
        ttft_mean   = perf.get("ttft_ms_mean")   or g("ttft_ms_mean",   "ttftMs")
        ttft_std    = perf.get("ttft_ms_stddev")  or g("ttft_ms_stddev", "ttftStdDev")
        tps_mean    = perf.get("decode_tokens_per_sec_mean") or g("decode_tokens_per_sec_mean", "decodeTokensPerSec")
        peak_mem    = perf.get("peak_memory_mb_mean") or g("peak_memory_mb_mean", "peakMemoryMB")
        on_disk     = perf.get("on_disk_size_mb") or g("on_disk_size_mb", "onDiskSizeMB")
        mmproj      = perf.get("mmproj_size_mb")

        if not model_key:
            print(f"  SKIP (no model_key): {path.name}")
            continue

        db.execute("""
            INSERT INTO iphone_perf VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (model_key, device_id, quant, runtime,
              ttft_mean, ttft_std, tps_mean, peak_mem, on_disk, mmproj,
              timestamp, exp_id, notes))
        rows += 1
    db.commit()
    return rows


def load_mac_quality(db: sqlite3.Connection) -> int:
    """Load Task 2.2 MetricsReport JSONs (quality scores only, no _config)."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS mac_quality (
            model_key   TEXT NOT NULL,
            device_id   TEXT NOT NULL,
            benchmark   TEXT NOT NULL,
            metric      TEXT NOT NULL,
            value       REAL,
            experiment_id TEXT
        )
    """)
    db.execute("DELETE FROM mac_quality")

    # Model key is the filename prefix (before first underscore + benchmark name)
    BENCHMARK_SUFFIXES = ("_POPE", "_RealWorldQA", "_MMBench_DEV_EN")

    rows = 0
    task22_dir = PROJECT_ROOT / "artifacts/eval_task_2_2_20260525_094121"
    if not task22_dir.exists():
        return 0

    for path in sorted(task22_dir.glob("*.json")):
        if "_config" in path.name:
            continue
        # Determine model_key and benchmark from filename
        stem = path.stem  # e.g. LFM2-VL-450M_POPE
        benchmark = None
        model_key = None
        for suf in BENCHMARK_SUFFIXES:
            if stem.endswith(suf):
                benchmark = suf.lstrip("_")
                model_key = stem[: -len(suf)]
                break
        if not model_key:
            continue

        # Skip the wrong MiniCPM run (4_5 → superseded by 4.6)
        if "4_5" in model_key:
            continue

        data = json.loads(path.read_text())
        for qs in data.get("quality_scores", []):
            db.execute("""
                INSERT INTO mac_quality VALUES (?,?,?,?,?,?)
            """, (
                model_key,
                data.get("device_id", "mac_mini_m4_16gb"),
                benchmark,
                qs.get("metric"),
                qs.get("value"),
                data.get("experiment_id"),
            ))
            rows += 1
    db.commit()
    return rows


def load_clip_scores(db: sqlite3.Connection) -> int:
    """Load per-model CLIP-score JSONs from artifacts/clip_scores/."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS clip_scores (
            model_key         TEXT NOT NULL,
            platform          TEXT,
            clip_model        TEXT,
            mean_clip_score   REAL,
            std_clip_score    REAL,
            n                 INTEGER,
            source_file       TEXT
        )
    """)
    db.execute("DELETE FROM clip_scores")

    # Platform labels inferred from filename
    PLATFORM_MAP = {
        "FastVLM-0.5B-iPhone-FP16": "iPhone 16 Pro (FP16 MLX)",
        "LFM2-VL-450M":             "Mac mini M4 (bfloat16)",
        "SmolVLM-500M":             "Mac mini M4 (bfloat16)",
        "MiniCPM-V-4.6":            "Mac mini M4 (bfloat16)",
    }

    rows = 0
    for path in sorted((PROJECT_ROOT / "artifacts/clip_scores").glob("*_clip.json")):
        data = json.loads(path.read_text())
        model_key = data.get("model_key", path.stem)
        platform  = PLATFORM_MAP.get(model_key, "unknown")
        db.execute("""
            INSERT INTO clip_scores VALUES (?,?,?,?,?,?,?)
        """, (
            model_key,
            platform,
            data.get("clip_model"),
            data.get("mean_clip_score"),
            data.get("std_clip_score"),
            data.get("n"),
            path.name,
        ))
        rows += 1
    db.commit()
    return rows


def load_clip_scores_n50(db: sqlite3.Connection) -> int:
    """Load the robust n=50 CLIP-score JSONs from artifacts/clip_scores_n50/ (Phase 2 P2-1.1)."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS clip_scores_n50 (
            model_key         TEXT NOT NULL,
            clip_model        TEXT,
            mean_clip_score   REAL,
            std_clip_score    REAL,
            n                 INTEGER,
            source_file       TEXT
        )
    """)
    db.execute("DELETE FROM clip_scores_n50")

    rows = 0
    src = PROJECT_ROOT / "artifacts/clip_scores_n50"
    if not src.exists():
        return 0
    for path in sorted(src.glob("*_clip.json")):
        data = json.loads(path.read_text())
        db.execute(
            "INSERT INTO clip_scores_n50 VALUES (?,?,?,?,?,?)",
            (
                data.get("model_key", path.stem),
                data.get("clip_model"),
                data.get("mean_clip_score"),
                data.get("std_clip_score"),
                data.get("n"),
                path.name,
            ),
        )
        rows += 1
    db.commit()
    return rows


def load_phase2_mcq(db: sqlite3.Connection) -> int:
    """Load Phase 2 P2-1.3 MCQ results from artifacts/phase2_mcq/.

    Each file is <model_key>_<benchmark>.json. Model keys distinguish the
    inference path: Qwen2.5-VL-3B (fp16 transformers), Qwen2.5-VL-3B-F16-GGUF,
    Qwen2.5-VL-3B-Q4_K_M (both via llama.cpp/mtmd). Lets the dashboard show the
    path-vs-quantization decomposition.
    """
    db.execute("""
        CREATE TABLE IF NOT EXISTS phase2_mcq (
            model_key     TEXT NOT NULL,
            benchmark     TEXT NOT NULL,
            metric        TEXT NOT NULL,
            value         REAL,
            experiment_id TEXT
        )
    """)
    db.execute("DELETE FROM phase2_mcq")

    BENCHMARK_SUFFIXES = ("_POPE", "_RealWorldQA", "_MMBench_DEV_EN")
    rows = 0
    src = PROJECT_ROOT / "artifacts/phase2_mcq"
    if not src.exists():
        return 0
    for path in sorted(src.glob("*.json")):
        if "_config" in path.name:
            continue
        stem = path.stem
        benchmark = model_key = None
        for suf in BENCHMARK_SUFFIXES:
            if stem.endswith(suf):
                benchmark = suf.lstrip("_")
                model_key = stem[: -len(suf)]
                break
        if not model_key:
            continue
        data = json.loads(path.read_text())
        for qs in data.get("quality_scores", []):
            db.execute(
                "INSERT INTO phase2_mcq VALUES (?,?,?,?,?)",
                (model_key, benchmark, qs.get("metric"), qs.get("value"),
                 data.get("experiment_id")),
            )
            rows += 1
    db.commit()
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build metrics SQLite DB (Phase 0 + Phase 2)")
    ap.add_argument("--db", default="metrics.db", help="Output SQLite file")
    args = ap.parse_args()

    db_path = PROJECT_ROOT / args.db
    db = sqlite3.connect(db_path)

    print(f"Building {db_path} …")
    n_iphone   = load_iphone_perf(db)
    n_quality  = load_mac_quality(db)
    n_clip     = load_clip_scores(db)
    n_clip50   = load_clip_scores_n50(db)
    n_p2_mcq   = load_phase2_mcq(db)

    print(f"  iphone_perf     : {n_iphone} rows")
    print(f"  mac_quality     : {n_quality} rows")
    print(f"  clip_scores     : {n_clip} rows")
    print(f"  clip_scores_n50 : {n_clip50} rows")
    print(f"  phase2_mcq      : {n_p2_mcq} rows")
    print(f"Done → {db_path}")
    db.close()


if __name__ == "__main__":
    main()
