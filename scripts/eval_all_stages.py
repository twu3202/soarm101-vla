"""Evaluate every available stage ckpt on its own num_active AND on n=5.

Useful at the end of the curriculum to see if earlier-stage policies still
generalize, and to compare best deterministic success across stages.
"""
from __future__ import annotations

import sys
import json
import pathlib
import argparse
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402
from env.sort_env import SortBlocksEnv  # noqa: E402
from env.state_wrappers import FlatStateWrapper  # noqa: E402


def _eval_ckpt(ckpt_path: pathlib.Path, num_active: int, n_episodes: int = 20,
               max_steps: int = 750) -> dict:
    if not ckpt_path.is_file():
        return {"error": "ckpt_missing"}
    model = PPO.load(str(ckpt_path), device="cuda")
    env = FlatStateWrapper(SortBlocksEnv(
        num_active_cubes=num_active, max_steps=max_steps,
        render_images=False, seed=999))
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        total_r = 0.0
        for t in range(max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            total_r += float(r)
            if term or trunc:
                break
        results.append({
            "success": bool(info.get("is_success", False)),
            "progress": int(info.get("sort_progress", 0)),
            "reward": total_r,
            "steps": t + 1,
        })
    env.close()
    succ = np.mean([r["success"] for r in results])
    prog = np.mean([r["progress"] for r in results])
    rew_m = np.mean([r["reward"] for r in results])
    rew_s = np.std([r["reward"] for r in results])
    return {
        "n_episodes": n_episodes,
        "success_rate": float(succ),
        "progress_mean": float(prog),
        "reward_mean": float(rew_m),
        "reward_std": float(rew_s),
        "episodes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--include-n5", action="store_true",
                        help="Also evaluate each stage's policy on n=5 task to see generalization")
    args = parser.parse_args()

    summary = {}
    for stage in (1, 2, 3, 4, 5):
        stage_dir = ROOT / "outputs" / f"ppo_stage{stage}"
        best_ckpt = stage_dir / "ckpts" / "best" / "best_model.zip"
        final_ckpt = stage_dir / "ckpts" / "final.zip"
        ckpt = best_ckpt if best_ckpt.is_file() else final_ckpt
        if not ckpt.is_file():
            summary[f"stage{stage}"] = {"status": "missing"}
            print(f"  stage {stage}: NO CKPT")
            continue
        print(f"\n=== stage {stage} ({ckpt.name}) ===")
        res = _eval_ckpt(ckpt, num_active=stage, n_episodes=args.n_episodes)
        del res["episodes"]  # too verbose
        summary[f"stage{stage}"] = {"native": res}
        print(f"  native (n={stage}): success={res['success_rate']:.1%}  "
              f"progress={res['progress_mean']:.2f}/{stage}  reward={res['reward_mean']:+.1f}±{res['reward_std']:.1f}")
        if args.include_n5 and stage < 5:
            res5 = _eval_ckpt(ckpt, num_active=5, n_episodes=args.n_episodes)
            del res5["episodes"]
            summary[f"stage{stage}"]["n5"] = res5
            print(f"  on n=5:     success={res5['success_rate']:.1%}  "
                  f"progress={res5['progress_mean']:.2f}/5  reward={res5['reward_mean']:+.1f}")

    out_path = ROOT / "outputs" / "all_stages_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
