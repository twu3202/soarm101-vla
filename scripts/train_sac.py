"""SB3 SAC trainer for the SO-ARM101 sort task.

PPO struggled with sparse-reward grasp discovery (policy mean stuck approaching but
not grasping). SAC's replay buffer should help re-use rare successful transitions.

Usage:
    python scripts/train_sac.py --stage 1 --total-steps 500_000
"""
from __future__ import annotations

import argparse
import sys
import time
import pathlib
import json
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback  # noqa: E402

from env.sort_env import SortBlocksEnv  # noqa: E402
from env.state_wrappers import FlatStateWrapper  # noqa: E402


def make_env(*, num_active_cubes, max_steps=750, seed=0, fixed_cube_xys=None,
             randomize_active_color=False, render_images=False, action_mode="absolute"):
    def _thunk():
        env = SortBlocksEnv(
            num_active_cubes=num_active_cubes, max_steps=max_steps,
            render_images=render_images, seed=seed,
            fixed_cube_xys=fixed_cube_xys,
            randomize_active_color=randomize_active_color,
            action_mode=action_mode,
        )
        env = FlatStateWrapper(env)
        return env
    return _thunk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4,
                        help="SAC uses fewer parallel envs since it's off-policy")
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=str, default="auto",
                        help="SAC entropy temperature (auto recommended)")
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--ckpt-every", type=int, default=100_000)
    parser.add_argument("--fixed-spawn", action="store_true",
                        help="Use fixed cube positions (easy-mode curriculum)")
    parser.add_argument("--random-color", action="store_true",
                        help="Randomize which color is active each episode (n=1 only); "
                             "helps the learned policy generalize to current_color_onehot != [1,0,0,0,0]")
    parser.add_argument("--action-mode", type=str, default="absolute", choices=["absolute", "delta"],
                        help="Action space: 'absolute' (default, works best at low n_envs) or 'delta' (ManiSkill style)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor. 0.99 default for sparse-reward; 0.8-0.9 for dense.")
    args = parser.parse_args()

    out = pathlib.Path(args.out) if args.out else (ROOT / "outputs" / f"sac_stage{args.stage}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tb").mkdir(exist_ok=True)
    (out / "ckpts").mkdir(exist_ok=True)
    eval_dir = out / "eval"
    eval_dir.mkdir(exist_ok=True)

    fixed = [(0.18, 0.05)] if args.fixed_spawn else None

    print(f"=== train_sac stage={args.stage} ===")
    print(f"  num_active: {args.stage}")
    print(f"  total_steps: {args.total_steps:,}")
    print(f"  buffer_size: {args.buffer_size:,}")
    print(f"  fixed_spawn: {fixed}")
    print(f"  out: {out}")

    train_env_fns = [make_env(num_active_cubes=args.stage, max_steps=args.max_steps,
                              seed=args.seed + i, fixed_cube_xys=fixed,
                              randomize_active_color=args.random_color,
                              action_mode=args.action_mode)
                     for i in range(args.n_envs)]
    train_env = SubprocVecEnv(train_env_fns) if args.n_envs > 1 else DummyVecEnv(train_env_fns)
    train_env = VecMonitor(train_env)

    eval_env_fns = [make_env(num_active_cubes=args.stage, max_steps=args.max_steps,
                             seed=1000 + i, fixed_cube_xys=fixed,
                             randomize_active_color=args.random_color,
                             action_mode=args.action_mode)
                    for i in range(3)]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecMonitor(eval_env, str(eval_dir / "monitor.csv"))

    if args.init_from is not None and pathlib.Path(args.init_from).is_file():
        print(f"  Loading from {args.init_from}")
        model = SAC.load(args.init_from, env=train_env, device="cuda")
        model.tensorboard_log = str(out / "tb")
    else:
        ent_coef = args.ent_coef
        try:
            ent_coef = float(ent_coef)
        except ValueError:
            pass
        model = SAC(
            "MlpPolicy", train_env,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            ent_coef=ent_coef,
            gamma=args.gamma, tau=0.005,
            train_freq=8, gradient_steps=8,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
            tensorboard_log=str(out / "tb"),
            verbose=1, device="cuda", seed=args.seed,
        )

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(out / "ckpts" / "best"),
        log_path=str(eval_dir),
        eval_freq=max(args.eval_every // args.n_envs, 1),
        n_eval_episodes=10, deterministic=True, render=False, verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.ckpt_every // args.n_envs, 1),
        save_path=str(out / "ckpts"), name_prefix="sac",
    )

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_steps,
                    callback=[eval_cb, ckpt_cb], log_interval=10)
    except KeyboardInterrupt:
        print("[interrupted]")
    finally:
        wall = time.time() - t0
        model.save(out / "ckpts" / "final.zip")
        with open(out / "training_summary.json", "w") as f:
            json.dump({"stage": args.stage, "total_steps": args.total_steps,
                       "wall_clock_min": wall / 60}, f, indent=2)
        train_env.close(); eval_env.close()
        print(f"[done] {args.total_steps:,} steps in {wall/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
