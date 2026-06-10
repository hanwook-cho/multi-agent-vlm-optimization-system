#!/usr/bin/env bash
# scripts/start_strategist_llm.sh — Launch the local LLM for the Search Strategist.
#
# Serves Qwen2.5-7B-Instruct via llama.cpp's server with native tool-calling
# enabled (--jinja). The Search Strategist talks to it over the OpenAI-compatible
# endpoint at http://localhost:8080/v1 (backend="llamacpp").
#
# Hardware: Qwen2.5-7B Q4_K_M (~5GB runtime) is sized for the M4 16GB Mac mini.
# On a 32GB machine, point MODEL at a qwen2.5-32b-instruct GGUF instead (ADR-0010).
#
# Prerequisites:
#   1. llama.cpp built with the server target:
#        git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
#        cmake -B build -DGGML_METAL=ON && cmake --build build --target llama-server -j
#   2. The GGUF model file. Download from HuggingFace, e.g.:
#        huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
#          Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir ./models
#
# Usage:
#   ./scripts/start_strategist_llm.sh                       # uses defaults below
#   MODEL=./models/Qwen2.5-32B-Instruct-Q4_K_M.gguf ./scripts/start_strategist_llm.sh
#   PORT=8090 ./scripts/start_strategist_llm.sh

set -euo pipefail

# ── Config (override via env) ──────────────────────────────────────────────
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"   # path to the llama-server binary
MODEL="${MODEL:-./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"            # context window; the strategist's prompt + ledger fits comfortably
NGL="${NGL:-999}"            # offload all layers to Metal GPU

# ── Pre-flight ──────────────────────────────────────────────────────────────
if ! command -v "$LLAMA_SERVER" >/dev/null 2>&1 && [ ! -x "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found (looked for '$LLAMA_SERVER')."
    echo "       Build it or set LLAMA_SERVER=/path/to/llama-server."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model file not found: $MODEL"
    echo "       Download a Qwen2.5 GGUF or set MODEL=/path/to/model.gguf"
    exit 1
fi

echo "── Starting llama-server for the Search Strategist ──────────────────────"
echo "  Model:   $MODEL"
echo "  Port:    $PORT   (endpoint: http://localhost:${PORT}/v1)"
echo "  Context: $CTX"
echo "  --jinja: enabled (native tool-calling for Qwen2.5)"
echo ""

# --jinja is REQUIRED: it enables the model's chat template so the OpenAI
# `tools` parameter is honored. Without it, Qwen2.5 falls back to ReAct mode.
exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    --port "$PORT" \
    -c "$CTX" \
    -ngl "$NGL" \
    --jinja
