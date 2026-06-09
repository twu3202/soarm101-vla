#!/bin/bash
# SO-101 cube-sort pi0.5 LoRA — full fine-tune (config num_train_steps=10000, batch 32).
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
LOG="$HOME/Projects/so101_sort/train_v1.log"
echo "############ SO101 pi0.5 LoRA v1 START $(date) PID=$$ ############" >> "$LOG"
uv run scripts/train.py pi05_so101_sort_lora \
    --exp-name=v1 \
    --no-wandb-enabled \
    --overwrite >> "$LOG" 2>&1
echo "############ SO101 pi0.5 LoRA v1 DONE  $(date) exit=$? ############" >> "$LOG"
