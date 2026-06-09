#!/bin/bash
set -e
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
python3 ~/Projects/so101_sort/patch_dataloader.py
echo "=== compute_norm_stats (pyav backend) ==="
uv run scripts/compute_norm_stats.py --config-name pi05_so101_sort_lora 2>&1 | tail -45
