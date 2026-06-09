"""E1 smoke test: load SortBlocksEnv, reset, render, step random actions, save frames.

Run from project root:
    python -m scripts.sanity_test
or:
    python scripts/sanity_test.py

PASS criteria:
- env constructs without error
- reset returns dict obs with expected keys + shapes
- 30 random-action steps run without crashing
- both camera frames saved to outputs/
"""
from __future__ import annotations

import os
import sys
import pathlib
import numpy as np

# Allow `python scripts/sanity_test.py` from project root
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from env.sort_env import SortBlocksEnv, CUBE_COLORS, TARGET_POSITIONS  # noqa: E402

OUT_DIR = ROOT / "outputs" / "sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_png(path: pathlib.Path, img: np.ndarray) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(str(path), img)
    except Exception as e:
        # Fallback: write raw via PIL if available
        from PIL import Image
        Image.fromarray(img).save(str(path))


def main() -> int:
    print("=== SortBlocksEnv smoke test ===")
    env = SortBlocksEnv(seed=0)
    print(f"  xml: {env._xml_path}")
    print(f"  action_space: {env.action_space}")
    print(f"  obs keys: {list(env.observation_space.spaces.keys())}")
    for k, sp in env.observation_space.spaces.items():
        print(f"    {k:18s} shape={sp.shape} dtype={sp.dtype}")
    print(f"  ctrl ranges (lo / hi):")
    for n, lo, hi in zip(env._arm_joint_names, env._ctrl_lo, env._ctrl_hi):
        print(f"    {n:14s} [{lo:+.3f}, {hi:+.3f}]")

    obs, info = env.reset(seed=42)
    print("\n[reset] OK")
    print(f"  state shape: {obs['state'].shape}, "
          f"cube_positions shape: {obs['cube_positions'].shape}")
    print(f"  initial cube_xy:\n{info['cube_xy']}")
    print(f"  target_xy:\n{info['target_xy']}")
    print(f"  initial sort_progress = {info['sort_progress']} / 5")
    print(f"  initial tip_xyz: {info['tip_xyz']}")

    # Save reset-state frames
    _save_png(OUT_DIR / "reset_front.png", obs["image_front"])
    _save_png(OUT_DIR / "reset_wrist.png", obs["image_wrist"])
    print(f"  saved reset frames → {OUT_DIR}/reset_{{front,wrist}}.png")

    # Step 30 random actions
    print("\n[stepping 30 random actions...]")
    total_r = 0.0
    rng = np.random.default_rng(0)
    for i in range(30):
        a = rng.uniform(-0.3, 0.3, size=6).astype(np.float32)  # small explorations
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if i in (0, 9, 19, 29):
            comp = info["reward_components"]
            print(f"  step {i:2d}: r={r:+.3f}  dist_sum={-comp['dist']:.3f}  "
                  f"in_target={comp['in_target_count']}  tip_z={info['tip_xyz'][2]:.3f}")
        if term or trunc:
            print(f"  early end at step {i+1} (term={term} trunc={trunc})")
            break

    print(f"\n[summary] total_reward over 30 steps = {total_r:.3f}")
    print(f"  final cube_xy:\n{info['cube_xy']}")
    print(f"  final cube_to_target_dist: {info['cube_to_target_dist']}")
    print(f"  final sort_progress = {info['sort_progress']} / 5")

    # Save final-state frames
    _save_png(OUT_DIR / "final_front.png", obs["image_front"])
    _save_png(OUT_DIR / "final_wrist.png", obs["image_wrist"])
    print(f"  saved final frames → {OUT_DIR}/final_{{front,wrist}}.png")

    env.close()
    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
