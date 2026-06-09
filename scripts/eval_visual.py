"""Visually evaluate a trained PPO ckpt — render an episode to PNG strip.

Usage:
    python scripts/eval_visual.py --ckpt outputs/ppo_stage1/ckpts/best/best_model.zip \
        --stage 1 --n-episodes 4
"""
from __future__ import annotations

import argparse
import sys
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402
from env.sort_env import SortBlocksEnv  # noqa: E402
from env.state_wrappers import FlatStateWrapper  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--n-episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out) if args.out else (ROOT / "outputs" / "ppo_eval_visual")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ckpt: {args.ckpt}")
    model = PPO.load(args.ckpt, device="cuda")

    # Render-enabled env
    env_inner = SortBlocksEnv(num_active_cubes=args.stage, max_steps=args.max_steps,
                              render_images=True, image_size=(280, 280), seed=42)
    env = FlatStateWrapper(env_inner)

    results = []
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=200 + ep)
        frames = []
        total_r = 0.0
        for t in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, term, trunc, info = env.step(action)
            total_r += float(r)
            if t % 6 == 0:
                frames.append(env_inner._render_cam(env_inner._cam_front_id))
            if term or trunc:
                # Record final frame
                frames.append(env_inner._render_cam(env_inner._cam_front_id))
                break
        success = bool(info.get("is_success", False))
        sort_progress = int(info.get("sort_progress", 0))
        print(f"  ep{ep}: success={success}  progress={sort_progress}/{args.stage}  "
              f"steps={t+1}  total_r={total_r:+.1f}")
        results.append({"ep": ep, "success": success, "progress": sort_progress,
                        "steps": t + 1, "total_r": total_r, "frames": frames})

    # Save strip per episode
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for r in results:
            frames = r["frames"]
            n = len(frames)
            cols = min(n, 8)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
            if rows == 1: axes = np.array([axes])
            if cols == 1: axes = axes.reshape(-1, 1)
            for i, f in enumerate(frames):
                ax = axes[i // cols, i % cols]
                ax.imshow(f); ax.set_title(f"t={i*6 if i<n-1 else r['steps']}", fontsize=7)
                ax.axis("off")
            for j in range(n, rows * cols):
                axes[j // cols, j % cols].axis("off")
            fig.suptitle(f"ep{r['ep']}  success={r['success']}  progress={r['progress']}/{args.stage}",
                         fontsize=10)
            fig.tight_layout()
            outp = out_dir / f"stage{args.stage}_ep{r['ep']}_success{int(r['success'])}.png"
            fig.savefig(outp, dpi=110)
            plt.close(fig)
            print(f"  saved {outp}")
    except ImportError:
        print("  matplotlib unavailable — skipped frames save")

    n_succ = sum(1 for r in results if r["success"])
    print(f"\n=== {n_succ}/{args.n_episodes} episodes succeeded ===")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
