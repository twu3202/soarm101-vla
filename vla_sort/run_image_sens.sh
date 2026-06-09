#!/bin/bash
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
pkill -f serve_policy 2>/dev/null
sleep 3
uv run python ~/Projects/so101_sort/image_sensitivity.py 2>&1 | grep -vE "Progress|it/s|Computing|warn|Warning|deprecat|INFO" | tail -25
