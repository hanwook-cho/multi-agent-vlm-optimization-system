#!/usr/bin/env bash
# scripts/retest_task_2_1.sh — Re-run Task 2.1 measurement (Qwen2.5-VL-3B on Mac mini).
#
# Usage:
#   ./scripts/retest_task_2_1.sh
#
# Before running:
#   - Quit all non-essential apps (browsers, Slack, etc.) to minimise swap.
#   - The model is already cached; this run skips the download.
#   - Results are written to results/ and copied to artifacts/.

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGES=(
    datasets/stage_a_proxy/photos/img1.jpg
    datasets/stage_a_proxy/photos/img2.jpg
    datasets/stage_a_proxy/photos/img3.jpg
    datasets/stage_a_proxy/photos/img4.jpg
    datasets/stage_a_proxy/photos/img5.jpg
)
OUTPUT="results/qwen25_vl_3b_fp16_mac_mini.json"

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo "── Pre-flight ───────────────────────────────────────────────────────────"

# Check all images exist
for img in "${IMAGES[@]}"; do
    if [[ ! -f "$img" ]]; then
        echo "ERROR: image not found: $img"
        echo "Run: curl -o $img <url>"
        exit 1
    fi
done

# Check available RAM
AVAIL_MB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().available / 1024 / 1024))")
SWAP_MB=$(python3 -c "import psutil; print(int(psutil.swap_memory().used / 1024 / 1024))")
echo "Available RAM: ${AVAIL_MB} MB"
echo "Swap in use:   ${SWAP_MB} MB"

if (( SWAP_MB > 3000 )); then
    echo ""
    echo "WARNING: ${SWAP_MB} MB of swap is in use (threshold: 3000 MB)."
    echo "Latency measurements will be swap_contaminated=True."
    echo "For clean numbers: quit other apps and rerun."
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
elif (( SWAP_MB > 0 )); then
    echo "Note: ${SWAP_MB} MB swap is macOS baseline compressed memory — acceptable."
    echo "The model runs entirely in MPS memory; this does not affect inference latency."
fi

mkdir -p results artifacts

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "── Measuring ────────────────────────────────────────────────────────────"
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 runners/measure_mac.py \
    --images "${IMAGES[@]}" \
    --output "$OUTPUT" \
    --offline

# ── Archive ──────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="artifacts/qwen25_vl_3b_fp16_mac_mini_${TIMESTAMP}.json"
cp "$OUTPUT" "$ARCHIVE"
cp "${OUTPUT%.json}_config.json" "${ARCHIVE%.json}_config.json"
echo ""
echo "── Archived ─────────────────────────────────────────────────────────────"
echo "  $ARCHIVE"

# ── Quick validation ──────────────────────────────────────────────────────────
echo ""
echo "── Validation ───────────────────────────────────────────────────────────"
python3 - <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from schemas import MetricsReport
r = MetricsReport.model_validate(json.loads(open("results/qwen25_vl_3b_fp16_mac_mini.json").read()))
print(f"  status:             {r.status}")
print(f"  device_id:          {r.device_id}")
print(f"  TTFT (median):      {r.ttft_ms:.0f} ms" if r.ttft_ms else "  TTFT: None")
print(f"  decode (median):    {r.decode_tokens_per_sec:.1f} tok/s" if r.decode_tokens_per_sec else "  decode: None")
print(f"  peak memory (MPS):  {r.peak_memory_mb:.0f} MB" if r.peak_memory_mb else "  peak memory: None")
print(f"  on-disk size:       {r.on_disk_size_mb:.0f} MB" if r.on_disk_size_mb else "  on-disk: None")
print(f"  swap_contaminated:  {r.hardware_fingerprint.swap_contaminated}")
PYEOF
