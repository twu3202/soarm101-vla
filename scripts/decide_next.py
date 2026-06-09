"""End-of-stage decision helper.

Reads a stage's training summary + eval history, decides whether to:
  (a) advance to next stage (success_rate ≥ MIN_OK)
  (b) extend training (success_rate ≥ MIN_PROGRESS, still trending up)
  (c) halt and report failure

Outputs JSON to stdout for piping into other tools.
"""
from __future__ import annotations

import sys
import json
import pathlib
import argparse
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

MIN_OK = 0.50         # success rate to advance
MIN_PROGRESS = 0.10   # below this AND not improving → halt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    args = parser.parse_args()

    stage_dir = ROOT / "outputs" / f"ppo_stage{args.stage}"
    eval_npz = stage_dir / "eval" / "evaluations.npz"

    if not eval_npz.is_file():
        print(json.dumps({"decision": "no_data", "stage": args.stage}))
        return 1

    d = np.load(eval_npz)
    # SB3 stores: timesteps, results (n_evals, n_episodes), ep_lengths,
    # and may store successes (n_evals, n_episodes) if env reports is_success
    timesteps = d["timesteps"] if "timesteps" in d.files else np.arange(len(d["results"]))
    if "successes" in d.files:
        success_per_eval = d["successes"].mean(axis=-1)  # (n_evals,)
    else:
        success_per_eval = np.zeros(len(timesteps))
    reward_per_eval = d["results"].mean(axis=-1)

    best_succ = float(success_per_eval.max()) if len(success_per_eval) > 0 else 0.0
    final_succ = float(success_per_eval[-1]) if len(success_per_eval) > 0 else 0.0
    last10 = success_per_eval[-10:] if len(success_per_eval) >= 10 else success_per_eval
    trending_up = bool(len(last10) >= 3 and last10[-1] > last10[0])

    decision = "halt"
    if best_succ >= MIN_OK:
        decision = "advance"
    elif final_succ >= MIN_PROGRESS or trending_up:
        decision = "extend"

    out = {
        "stage": args.stage,
        "decision": decision,
        "n_evals": int(len(timesteps)),
        "best_success_rate": best_succ,
        "final_success_rate": final_succ,
        "best_reward": float(reward_per_eval.max()),
        "final_reward": float(reward_per_eval[-1]),
        "trending_up": trending_up,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
