#!/bin/bash
# Serve the trained SO-101 pi0.5 LoRA policy over websocket (default port 8000).
# Usage:  bash ~/Projects/so101_sort/serve_so101.sh [STEP]
#   STEP defaults to the latest numeric checkpoint under checkpoints/pi05_so101_sort_lora/v1/.
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

CKPT_ROOT=checkpoints/pi05_so101_sort_lora/v1
if [ -n "$1" ]; then
    STEP="$1"
else
    STEP=$(ls "$CKPT_ROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
fi
DIR="$CKPT_ROOT/$STEP"
if [ ! -d "$DIR" ]; then
    echo "ERROR: checkpoint dir not found: $DIR"; echo "available:"; ls "$CKPT_ROOT" 2>/dev/null; exit 1
fi

echo "Serving checkpoint: $DIR  (port 8000)"
uv run scripts/serve_policy.py \
    --port 8000 \
    --default-prompt "Sort the cubes into a row left to right: red, orange, yellow, green, blue." \
    policy:checkpoint \
    --policy.config pi05_so101_sort_lora \
    --policy.dir "$DIR"
