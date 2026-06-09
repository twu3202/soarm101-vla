"""Sequentially train PPO across curriculum stages 1→5.

Each stage warm-starts from the previous stage's best ckpt. If a stage fails to
reach the target success rate, training stops (rather than rolling forward and
wasting compute on a broken policy).
"""
from __future__ import annotations

import argparse
import sys
import subprocess
import json
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# Default budgets per stage. Earlier stages are easier (smaller search space) but
# also "cold-start" from scratch is harder than warm-starting from a similar task.
DEFAULT_STEPS = {
    1: 3_000_000,
    2: 2_000_000,
    3: 2_000_000,
    4: 2_000_000,
    5: 3_000_000,
}


def _read_eval_success(stage_dir: pathlib.Path) -> float:
    """Return the best success rate seen during eval for this stage."""
    eval_path = stage_dir / "eval" / "evaluations.npz"
    if not eval_path.is_file():
        return 0.0
    try:
        import numpy as np
        d = np.load(eval_path)
        if "successes" in d.files:
            return float(d["successes"].mean(axis=-1).max())
        # Fall back to reward-based proxy
        return -1.0
    except Exception:
        return 0.0


def _run_stage(stage: int, steps: int, init_from: pathlib.Path | None,
               n_envs: int = 8, py: str = None) -> pathlib.Path:
    py = py or r"C:\Users\asus\miniconda3\envs\lerobot\python.exe"
    out_dir = ROOT / "outputs" / f"ppo_stage{stage}"
    print(f"\n=== STAGE {stage} ===")
    print(f"  output: {out_dir}")
    print(f"  init_from: {init_from}")
    print(f"  total_steps: {steps:,}")

    cmd = [
        py, str(ROOT / "scripts" / "train_ppo.py"),
        "--stage", str(stage),
        "--total-steps", str(steps),
        "--n-envs", str(n_envs),
        "--eval-every", "50000",
        "--ckpt-every", "100000",
        "--out", str(out_dir),
    ]
    if init_from is not None:
        cmd += ["--init-from", str(init_from)]

    log_path = out_dir.with_suffix(".log")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  log: {log_path}")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    dt = time.time() - t0
    print(f"  stage {stage} done in {dt/60:.1f} min (exit={proc.returncode})")

    success_rate = _read_eval_success(out_dir)
    print(f"  best eval success rate: {success_rate:.2%}")
    return out_dir / "ckpts" / "best" / "best_model.zip"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--end", type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--init", type=str, default=None,
                        help="Path to ckpt for stage --start init (skipped if stage 1)")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None,
                        help="Override default per-stage step budget")
    parser.add_argument("--min-success", type=float, default=0.50,
                        help="Required success rate to advance to next stage")
    args = parser.parse_args()

    init = pathlib.Path(args.init) if args.init else None
    for stage in range(args.start, args.end + 1):
        steps = args.steps if args.steps is not None else DEFAULT_STEPS.get(stage, 3_000_000)
        best_ckpt = _run_stage(stage, steps, init_from=init, n_envs=args.n_envs)
        if not best_ckpt.is_file():
            print(f"  WARNING: no best ckpt at {best_ckpt} — using final.zip")
            best_ckpt = ROOT / "outputs" / f"ppo_stage{stage}" / "ckpts" / "final.zip"
        success = _read_eval_success(ROOT / "outputs" / f"ppo_stage{stage}")
        if success < args.min_success:
            print(f"\n[STOP] stage {stage} success={success:.2%} < min={args.min_success:.2%}; "
                  f"halting curriculum so we don't compound failure into later stages.")
            return 1
        init = best_ckpt

    print("\n=== curriculum complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
