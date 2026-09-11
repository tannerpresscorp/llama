#!/usr/bin/env bash
set -euo pipefail
LLAMA="${LLAMA:-llama}"
MODEL="${MODEL:-artifacts/tiny-llama-f16.gguf}"
PORT="${PORT:-8080}"
exec "$LLAMA" serve -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c 1024
