#!/bin/bash
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "=== micro-smoke: 5 steps, batch 8  ($(date)) ==="
uv run scripts/train.py pi05_so101_sort_lora \
    --exp-name=v1_smoke \
    --num-train-steps=5 \
    --batch-size=8 \
    --no-wandb-enabled \
    --overwrite 2>&1 | tail -60
echo "=== smoke train.py exit code: ${PIPESTATUS[0]}  ($(date)) ==="
