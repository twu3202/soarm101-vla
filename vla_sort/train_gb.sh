#!/bin/bash
# Green-cube-into-bowl, 2-camera pi0.5 LoRA — full fine-tune (4000 steps, batch 32, keep every 1000).
cd ~/Projects/openpi
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
LOG="$HOME/Projects/so101_sort/train_gb.log"
echo "############ green_bowl pi0.5 LoRA v1 START $(date) PID=$$ ############" >> "$LOG"
uv run scripts/train.py pi05_green_bowl_lora --exp-name=v1 --num-train-steps=4000 --keep-period=1000 --no-wandb-enabled --overwrite >> "$LOG" 2>&1
echo "############ green_bowl pi0.5 LoRA v1 DONE  $(date) exit=$? ############" >> "$LOG"
