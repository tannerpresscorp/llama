#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-${LLAMA:-llama}}"
MODEL_PATH="${LLAMA_MODEL:-${MODEL:-artifacts/tiny-llama-f16.gguf}}"
HOST="${LLAMA_HOST:-${HOST:-127.0.0.1}}"
PORT="${LLAMA_PORT:-${PORT:-8080}}"
CONTEXT_SIZE="${LLAMA_CONTEXT_SIZE:-${CONTEXT_SIZE:-1024}}"

if ! command -v "$LLAMA_BIN" >/dev/null 2>&1; then
  echo "Could not find llama.cpp server binary: $LLAMA_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Could not find GGUF model file: $MODEL_PATH" >&2
  exit 1
fi

echo "Starting llama.cpp server on ${HOST}:${PORT} with model ${MODEL_PATH}"
exec "$LLAMA_BIN" serve -m "$MODEL_PATH" --host "$HOST" --port "$PORT" -c "$CONTEXT_SIZE"
