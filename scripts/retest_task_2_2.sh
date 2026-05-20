#!/usr/bin/env bash
# scripts/retest_task_2_2.sh — Re-run Task 2.2 quality evaluation.
#
# Runs all 5 models × 3 benchmarks × 100 examples and archives results.
# Expected runtime: 3-6 hours total (varies by model size).
#
# Usage:
#   ./scripts/retest_task_2_2.sh                     # all models, all benchmarks
#   ./scripts/retest_task_2_2.sh --models Qwen2.5-VL-3B --benchmarks POPE

set -euo pipefail
cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="results/eval_task_2_2"
ARCHIVE_DIR="artifacts/eval_task_2_2_${TIMESTAMP}"

# ── Pre-flight ─────────────────────────────────────────────────────────────
echo "── Pre-flight ────────────────────────────────────────────────────────"
AVAIL_MB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().available / 1024 / 1024))")
SWAP_MB=$(python3 -c "import psutil; print(int(psutil.swap_memory().used / 1024 / 1024))")
echo "Available RAM: ${AVAIL_MB} MB"
echo "Swap in use:   ${SWAP_MB} MB"

if (( SWAP_MB > 3000 )); then
    echo "WARNING: high swap (${SWAP_MB} MB). Consider quitting other apps."
    read -p "Continue? [y/N] " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

mkdir -p "$OUTPUT" "$ARCHIVE_DIR"

# ── Run ────────────────────────────────────────────────────────────────────
echo ""
echo "── Running eval ──────────────────────────────────────────────────────"
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 runners/eval_vlmeval.py \
    --output "$OUTPUT" \
    "$@"

# ── Archive ────────────────────────────────────────────────────────────────
cp "$OUTPUT"/*.json "$ARCHIVE_DIR"/ 2>/dev/null || true
echo ""
echo "── Archived to $ARCHIVE_DIR ──────────────────────────────────────────"
ls "$ARCHIVE_DIR"/*.json 2>/dev/null | sed 's|.*/||'

# ── Validate all reports ───────────────────────────────────────────────────
echo ""
echo "── Validation ────────────────────────────────────────────────────────"
python3 - "$OUTPUT" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from schemas import MetricsReport
outdir = Path(sys.argv[1])
reports = sorted(outdir.glob("*.json"))
reports = [p for p in reports if not p.name.endswith("_config.json")]
if not reports:
    print("  No report files found.")
    sys.exit(1)
print(f"  {'File':<45} {'Status':<12} {'Scores'}")
print("  " + "-" * 80)
for p in reports:
    r = MetricsReport.model_validate(json.loads(p.read_text()))
    scores_str = "  ".join(
        f"{s.metric}={s.value:.3f}" for s in r.quality_scores
    )
    print(f"  {p.name:<45} {r.status:<12} {scores_str}")
PYEOF
