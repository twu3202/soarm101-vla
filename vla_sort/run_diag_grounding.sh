#!/bin/bash
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "stopping any running policy server to free GPU..."
pkill -f serve_policy 2>/dev/null
sleep 3
echo "=== teacher-forcing grounding diagnostic ==="
uv run python ~/Projects/so101_sort/diag_grounding.py 2>&1 | grep -vE "Progress|it/s|Computing|warn|Warning|deprecat" | tail -40
