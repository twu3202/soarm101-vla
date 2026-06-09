"""SB3 PPO trainer for the SO-ARM101 sort task.

Usage:
    python scripts/train_ppo.py --stage 1 --total-steps 3_000_000

Curriculum:
    stage k = num_active_cubes = k.  Stages can warm-start from a prior ckpt via
    --init-from <path>.

All cameras are disabled (state-only) for >10x speedup. Re-enable in Phase 3.
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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback  # noqa: E402

from env.sort_env import SortBlocksEnv  # noqa: E402
from env.state_wrappers import FlatStateWrapper  # noqa: E402


def make_env(*, num_active_cubes: int, max_steps: int = 750, seed: int = 0,
             render_images: bool = False):
    def _thunk():
        env = SortBlocksEnv(
            num_active_cubes=num_active_cubes,
            max_steps=max_steps,
            render_images=render_images,
            seed=seed,
        )
        env = FlatStateWrapper(env)
        return env
    return _thunk


class SortProgressCallback(BaseCallback):
    """Logs sort_progress, success rate, and key reward components."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_rewards: list[float] = []
        self._ep_progress: list[int] = []
        self._ep_success: list[bool] = []
        self._ep_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            ep = info.get("episode") if isinstance(info, dict) else None
            sp = info.get("sort_progress", None) if isinstance(info, dict) else None
            success = bool(info.get("is_success", False)) if isinstance(info, dict) else False
            if ep is not None:
                self._ep_rewards.append(float(ep.get("r", 0)))
            if sp is not None:
                self._ep_progress.append(int(sp))
            self._ep_success.append(success)
            self._ep_count += 1
        # Every 50 episodes, log to tensorboard
        if self._ep_count >= 50 and len(self._ep_rewards) > 0:
            arr_r = np.array(self._ep_rewards[-50:])
            arr_p = np.array(self._ep_progress[-50:]) if self._ep_progress else np.array([0])
            arr_s = np.array(self._ep_success[-50:], dtype=np.float32)
            self.logger.record("rollout/ep_reward_mean", float(arr_r.mean()))
            self.logger.record("rollout/ep_progress_mean", float(arr_p.mean()))
            self.logger.record("rollout/ep_success_rate", float(arr_s.mean()))
            self._ep_count = 0
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--total-steps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to .zip ckpt to warm-start from (for curriculum)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: outputs/ppo_stage<k>)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout horizon per env")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--log-std-init", type=float, default=-1.0,
                        help="Initial log_std for the Gaussian policy. -1 → σ≈0.37 (more committed than SB3 default of 0).")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--ckpt-every", type=int, default=100_000)
    parser.add_argument("--use-dummy", action="store_true",
                        help="Use DummyVecEnv (single-process) instead of SubprocVecEnv")
    args = parser.parse_args()

    out = pathlib.Path(args.out) if args.out else (ROOT / "outputs" / f"ppo_stage{args.stage}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tb").mkdir(exist_ok=True)
    (out / "ckpts").mkdir(exist_ok=True)
    eval_dir = out / "eval"
    eval_dir.mkdir(exist_ok=True)

    print(f"=== train_ppo stage={args.stage} ===")
    print(f"  num_active_cubes  : {args.stage}")
    print(f"  n_envs            : {args.n_envs}")
    print(f"  total_steps       : {args.total_steps:,}")
    print(f"  init_from         : {args.init_from}")
    print(f"  out               : {out}")

    VecCls = DummyVecEnv if args.use_dummy else SubprocVecEnv
    train_env_fns = [make_env(num_active_cubes=args.stage, max_steps=args.max_steps,
                              seed=args.seed + i) for i in range(args.n_envs)]
    train_env = VecCls(train_env_fns)
    train_env = VecMonitor(train_env)

    # Eval env: 4 fixed seeds, single process
    eval_env_fns = [make_env(num_active_cubes=args.stage, max_steps=args.max_steps,
                             seed=1000 + i) for i in range(4)]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecMonitor(eval_env, str(eval_dir / "monitor.csv"))

    if args.init_from is not None:
        print(f"  Loading from {args.init_from}")
        model = PPO.load(args.init_from, env=train_env, device="cuda")
        # tensorboard log dir
        model.tensorboard_log = str(out / "tb")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=10,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               log_std_init=args.log_std_init),
            tensorboard_log=str(out / "tb"),
            verbose=1,
            device="cuda",
            seed=args.seed,
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out / "ckpts" / "best"),
        log_path=str(eval_dir),
        eval_freq=max(args.eval_every // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.ckpt_every // args.n_envs, 1),
        save_path=str(out / "ckpts"),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    progress_cb = SortProgressCallback()

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=[eval_cb, ckpt_cb, progress_cb],
            progress_bar=False,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("\n[interrupted — saving current state]")
    finally:
        wall = time.time() - t0
        model.save(out / "ckpts" / "final.zip")
        with open(out / "training_summary.json", "w") as f:
            json.dump({
                "stage": args.stage,
                "total_steps": args.total_steps,
                "wall_clock_sec": wall,
                "wall_clock_min": wall / 60,
                "steps_per_sec": args.total_steps / wall if wall > 0 else 0,
            }, f, indent=2)
        train_env.close()
        eval_env.close()
        print(f"\n[done] {args.total_steps:,} steps in {wall/60:.1f} min "
              f"= {args.total_steps/wall:.0f} steps/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
