#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_DIR:?Set LLAMA_DIR}"
: "${MODEL:?Set MODEL}"
: "${PROMPT_FILE:?Set PROMPT_FILE}"
: "${CACHE_PLAN:?Set CACHE_PLAN}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/reproduction-output}"
PROMPT_TOKENS="${PROMPT_TOKENS:-16363}"
CONTEXT_SIZE="${CONTEXT_SIZE:-17408}"
THREADS="${THREADS:-24}"
UBATCH="${UBATCH:-8192}"

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"
cmake -G Ninja -S "$ROOT/tools" -B "$BUILD_DIR" \
  -DLLAMA_CPP_DIR="$LLAMA_DIR" \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j

PREFIX_RUNNER="$BUILD_DIR/slotstream-prefix-cache"
DECODE_RUNNER="$BUILD_DIR/slotstream-hybrid-decode"
STATE="$OUTPUT_DIR/session.bin"

"$PREFIX_RUNNER" \
  -m "$MODEL" \
  --prompt-file "$PROMPT_FILE" \
  --state-file "$STATE" \
  --output-json "$OUTPUT_DIR/prefill-roundtrip.json" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --measured-tokens 1 \
  --verify-tokens 8 \
  -c "$CONTEXT_SIZE" \
  -t "$THREADS" \
  --n-batch "$UBATCH" \
  --n-ubatch "$UBATCH"

for run in 1 2 3; do
  "$DECODE_RUNNER" \
    -m "$MODEL" \
    --state-file "$STATE" \
    --output-json "$OUTPUT_DIR/decode-run-$run.json" \
    --hybrid-cache-plan "$CACHE_PLAN" \
    --promote-threshold -1 \
    --verify-tokens 1 \
    --warmup-tokens 100 \
    --measured-tokens 500 \
    -c "$CONTEXT_SIZE" \
    -t "$THREADS" \
    --n-batch 2048 \
    --n-ubatch 512
done

printf 'Results: %s\n' "$OUTPUT_DIR"
