#!/bin/bash
# SO-ARM101 cube-sort -> pi0.5 LoRA fine-tune on a single 48GB GPU.
# Detached launch (survives SSH disconnect):
#     nohup setsid bash ~/Projects/so101_sort/run_so101.sh </dev/null >/dev/null 2>&1 &
# Progress: ~/Projects/so101_sort/train.log   (tail -f it)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd "$HOME/Projects/openpi"

PKG="$HOME/Projects/so101_sort"
mkdir -p "$PKG"
exec >> "$PKG/train.log" 2>&1

CFG=pi05_so101_sort_lora
EXP=v1

echo
echo "######## SO101 pi0.5 LoRA run $(date) PID=$$ ########"

# 1) Norm stats for OUR dataset (required for a new dataset; written to assets/<repo_id>).
echo "=== [1/3] compute_norm_stats ($(date)) ==="
uv run scripts/compute_norm_stats.py "$CFG"

# 2) 5-step micro-smoke through the REAL data pipeline — catches key/shape wiring bugs
#    in minutes instead of after hours of GPU.
echo "=== [2/3] micro-smoke 5 steps, batch 8 ($(date)) ==="
uv run scripts/train.py "$CFG" \
    --exp-name="${EXP}_smoke" \
    --num-train-steps=5 \
    --batch-size=8 \
    --no-wandb-enabled \
    --overwrite

# 3) Real LoRA fine-tune.
echo "=== [3/3] train ($(date)) ==="
uv run scripts/train.py "$CFG" \
    --exp-name="$EXP" \
    --no-wandb-enabled \
    --overwrite

echo "######## DONE $(date) — checkpoints: checkpoints/$CFG/$EXP ########"
