#!/bin/bash
# Run the green-bowl offline validation (teacher-forcing + camera ablation) for one checkpoint step.
# Usage: bash ~/Projects/so101_sort/run_diag_gb.sh [STEP]   (default 4000)
# NOTE: needs the GPU free -> only run AFTER training has stopped.
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
STEP="${1:-4000}"
echo "=== green-bowl validation, ckpt step $STEP ==="
uv run python ~/Projects/so101_sort/diag_gb.py "$STEP" 2>&1 | grep -vE "Progress|it/s|Computing|warn|Warning|deprecat|INFO|UserWarning|torchvision" | tail -40
