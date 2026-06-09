"""SB3 PPO trainer with images (96x96) + state — Phase 3 (E4).

Initializes from the final state-only ckpt where possible. CNN extracts from
front + wrist cams, and is concatenated with the privileged state for the policy
input. Heavy domain randomization is OFF initially — we want to confirm vision
learning works before adding noise.

Usage:
    python scripts/train_ppo_vision.py --init-from outputs/ppo_stage5/ckpts/best/best_model.zip \
                                       --total-steps 10_000_000
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

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.torch_layers import CombinedExtractor  # noqa: E402

from env.sort_env import SortBlocksEnv  # noqa: E402
from env.multi_input_wrapper import VisionPPOWrapper  # noqa: E402


def make_env(*, num_active_cubes: int, max_steps: int, seed: int,
             image_size: tuple[int, int] = (96, 96)):
    def _thunk():
        env = SortBlocksEnv(
            num_active_cubes=num_active_cubes, max_steps=max_steps,
            render_images=True, image_size=image_size, seed=seed,
        )
        env = VisionPPOWrapper(env, include_wrist=True)
        return env
    return _thunk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-active", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=6,
                        help="Lower than state-only — image rendering bottlenecks throughput")
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--image-h", type=int, default=96)
    parser.add_argument("--image-w", type=int, default=96)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--ckpt-every", type=int, default=200_000)
    parser.add_argument("--use-dummy", action="store_true")
    args = parser.parse_args()

    out = pathlib.Path(args.out) if args.out else (ROOT / "outputs" / "ppo_vision")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tb").mkdir(exist_ok=True)
    (out / "ckpts").mkdir(exist_ok=True)
    eval_dir = out / "eval"
    eval_dir.mkdir(exist_ok=True)

    print(f"=== train_ppo_vision (num_active={args.num_active}) ===")
    print(f"  image_size: {args.image_h}x{args.image_w}")
    print(f"  n_envs: {args.n_envs}")
    print(f"  total_steps: {args.total_steps:,}")
    print(f"  init_from: {args.init_from}")
    print(f"  out: {out}")

    VecCls = DummyVecEnv if args.use_dummy else SubprocVecEnv
    train_env_fns = [make_env(num_active_cubes=args.num_active, max_steps=args.max_steps,
                              seed=args.seed + i,
                              image_size=(args.image_h, args.image_w))
                     for i in range(args.n_envs)]
    train_env = VecCls(train_env_fns)
    train_env = VecMonitor(train_env)

    eval_env_fns = [make_env(num_active_cubes=args.num_active, max_steps=args.max_steps,
                             seed=1000 + i,
                             image_size=(args.image_h, args.image_w))
                    for i in range(3)]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecMonitor(eval_env, str(eval_dir / "monitor.csv"))

    if args.init_from is not None and pathlib.Path(args.init_from).is_file():
        print(f"  Loading state-only ckpt from {args.init_from}")
        # Cannot directly load state-only policy weights into multi-input policy.
        # Instead we initialize fresh and let CNN co-train with the state branch.
        # Future work: distill the state-only behavior into the vision policy via DAgger.
        print("  (note: state-only ckpt structure differs from MultiInputPolicy; "
              "vision policy is fresh; state-only ckpt only used as a behavioral reference)")
        model = PPO(
            "MultiInputPolicy", train_env,
            n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=args.lr, ent_coef=args.ent_coef,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
            max_grad_norm=0.5, n_epochs=10,
            tensorboard_log=str(out / "tb"),
            verbose=1, device="cuda", seed=args.seed,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                features_extractor_class=CombinedExtractor,
                features_extractor_kwargs=dict(cnn_output_dim=128),
            ),
        )
    else:
        model = PPO(
            "MultiInputPolicy", train_env,
            n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=args.lr, ent_coef=args.ent_coef,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
            max_grad_norm=0.5, n_epochs=10,
            tensorboard_log=str(out / "tb"),
            verbose=1, device="cuda", seed=args.seed,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                features_extractor_class=CombinedExtractor,
                features_extractor_kwargs=dict(cnn_output_dim=128),
            ),
        )

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(out / "ckpts" / "best"),
        log_path=str(eval_dir),
        eval_freq=max(args.eval_every // args.n_envs, 1),
        n_eval_episodes=6, deterministic=True, render=False, verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.ckpt_every // args.n_envs, 1),
        save_path=str(out / "ckpts"), name_prefix="ppo_vision",
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
            json.dump({"total_steps": args.total_steps,
                       "wall_clock_min": wall / 60,
                       "steps_per_sec": args.total_steps / wall if wall > 0 else 0}, f, indent=2)
        train_env.close(); eval_env.close()
        print(f"[done] {args.total_steps:,} steps in {wall/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
