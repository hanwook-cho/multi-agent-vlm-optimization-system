#!/usr/bin/env bash
# scripts/convert_qwen25vl_gguf.sh — Phase 2 P2-1.2
# Convert Qwen2.5-VL-3B-Instruct (HF) → GGUF for llama.cpp/mtmd, quantize LM to Q4_K_M.
#
# Produces a deployable multimodal bundle (text LM + vision mmproj):
#   models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Q4_K_M.gguf   (text backbone, ~1.93GB)
#   models/qwen2.5-vl-3b-gguf/mmproj-Qwen2.5-VL-3B-f16.gguf (vision projector, ~1.34GB)
#
# The mmproj stays F16: sub-Q8_0 mmproj quantization is blocked with current
# tooling (Phase 1 H004). The vision encoder is the bulk of the on-disk size.
#
# NOTE (Qwen-VL): the model wants >=1024 image tokens for grounding accuracy
# (llama.cpp warns and suggests --image-min-tokens 1024). That large vision-token
# count drives TTFT on-device — relevant to the P2-1.4 iPhone feasibility gate.
#
# Prereqs: llama.cpp built (convert_hf_to_gguf.py + build/bin/llama-quantize),
#          Qwen2.5-VL-3B-Instruct in the HF cache (Phase 0 Task 2.1).
#
# Usage:  ./scripts/convert_qwen25vl_gguf.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LLAMACPP="vendor/llama.cpp"
OUTDIR="models/qwen2.5-vl-3b-gguf"
mkdir -p "$OUTDIR"

SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*/ | head -1)
if [ -z "${SNAP:-}" ] || [ ! -f "${SNAP}config.json" ]; then
    echo "ERROR: Qwen2.5-VL-3B-Instruct not found in HF cache."
    echo "       Fetch it first (it was cached in Phase 0 Task 2.1)."
    exit 1
fi
echo "Source snapshot: $SNAP"

echo "── Step 1/3: text backbone → F16 GGUF ──"
python3 "$LLAMACPP/convert_hf_to_gguf.py" "$SNAP" \
    --outtype f16 \
    --outfile "$OUTDIR/Qwen2.5-VL-3B-f16.gguf"

echo "── Step 2/3: vision projector → mmproj F16 GGUF ──"
python3 "$LLAMACPP/convert_hf_to_gguf.py" "$SNAP" \
    --mmproj \
    --outfile "$OUTDIR/mmproj-Qwen2.5-VL-3B-f16.gguf"

echo "── Step 3/3: quantize text backbone → Q4_K_M ──"
"$LLAMACPP/build/bin/llama-quantize" \
    "$OUTDIR/Qwen2.5-VL-3B-f16.gguf" \
    "$OUTDIR/Qwen2.5-VL-3B-Q4_K_M.gguf" \
    Q4_K_M

echo ""
echo "── Done. Deployable bundle: ──"
ls -lah "$OUTDIR/Qwen2.5-VL-3B-Q4_K_M.gguf" "$OUTDIR/mmproj-Qwen2.5-VL-3B-f16.gguf"
echo ""
echo "Verify:  $LLAMACPP/build/bin/llama-mtmd-cli \\"
echo "           -m $OUTDIR/Qwen2.5-VL-3B-Q4_K_M.gguf \\"
echo "           --mmproj $OUTDIR/mmproj-Qwen2.5-VL-3B-f16.gguf \\"
echo "           --image datasets/stage_a/photos/000000006040.jpg \\"
echo "           -p 'Describe what you see in this image.' -n 60"
