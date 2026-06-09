"""Print tip site position for each keyframe + a grid of candidate poses.

This is a one-shot diagnostic. The goal is to find joint settings that put the
gripper TIP at a known xyz so we can hand-craft a working grasp oracle.
"""
from __future__ import annotations

import sys
import pathlib
import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from env.sort_env import SortBlocksEnv, CUBE_REST_Z  # noqa: E402


def tip_for_qpos(env: SortBlocksEnv, q_arm: list[float]) -> np.ndarray:
    env.data.qpos[env._arm_qpos_adr] = np.asarray(q_arm)
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    return env.data.site_xpos[env._tip_site_id].copy()


def main() -> int:
    env = SortBlocksEnv(num_active_cubes=1, render_images=False, seed=0)
    env.reset(seed=0)

    print("=== keyframe tip positions (current grasp_test.py keyframes) ===")
    KEYFRAMES = {
        "home_open":           [ 0.00, -0.80,  1.20, -0.40,  0.00,  1.60],
        "above_cube_open":     [ 0.15, -0.40,  1.30, -1.10,  0.00,  1.60],
        "at_cube_open":        [ 0.15, -0.05,  1.30, -1.50,  0.00,  1.60],
        "at_cube_closed":      [ 0.15, -0.05,  1.30, -1.50,  0.00, -0.10],
        "lifted_closed":       [ 0.15, -0.80,  1.20, -0.40,  0.00, -0.10],
        "above_slot_closed":   [-0.30, -0.80,  1.20, -0.40,  0.00, -0.10],
        "at_slot_closed":      [-0.30, -0.05,  1.30, -1.50,  0.00, -0.10],
    }
    for name, q in KEYFRAMES.items():
        tip = tip_for_qpos(env, q)
        print(f"  {name:22s} q={q}  →  tip xyz = {tip}")
    print("\n  cube spawn target: (0.18, 0.05, 0.014)")
    print("  red slot target:   (0.30, -0.10)")

    # Scan a grid of shoulder_lift and elbow_flex values to find the pose with
    # tip near (0.18, 0.05, 0.014). shoulder_pan and wrist_flex constrained so
    # the gripper is "pointing down" at the table.
    print("\n=== pose grid scan: looking for tip ≈ (0.18, 0.05, 0.05) (5 cm above cube) ===")
    target = np.array([0.18, 0.05, 0.05])
    best = None
    best_d = 1e9
    for shoulder_pan in np.linspace(-0.3, 0.7, 6):
        for shoulder_lift in np.linspace(-1.5, 1.5, 9):
            for elbow_flex in np.linspace(-1.5, 1.5, 9):
                for wrist_flex in np.linspace(-1.5, 1.5, 9):
                    q = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, 0.0, 1.6]
                    tip = tip_for_qpos(env, q)
                    if tip[2] < 0.01: continue  # below table — skip
                    d = float(np.linalg.norm(tip - target))
                    if d < best_d:
                        best_d, best = d, (q, tip)
    print(f"  best q for tip ≈ {target} : q={best[0]}  →  tip={best[1]}  (d={best_d:.4f})")

    # Then refine for "at cube" (z = 0.014)
    print("\n=== refine: looking for tip ≈ (0.18, 0.05, 0.02) (just above cube top) ===")
    target = np.array([0.18, 0.05, 0.02])
    best2 = None
    best_d2 = 1e9
    q0 = best[0]
    for shoulder_lift in np.linspace(q0[1] - 0.5, q0[1] + 0.5, 11):
        for elbow_flex in np.linspace(q0[2] - 0.5, q0[2] + 0.5, 11):
            for wrist_flex in np.linspace(q0[3] - 0.8, q0[3] + 0.8, 11):
                q = [q0[0], shoulder_lift, elbow_flex, wrist_flex, 0.0, 1.6]
                tip = tip_for_qpos(env, q)
                d = float(np.linalg.norm(tip - target))
                if d < best_d2:
                    best_d2, best2 = d, (q, tip)
    print(f"  best q for tip ≈ {target} : q={best2[0]}  →  tip={best2[1]}  (d={best_d2:.4f})")

    # Above red slot (0.30, -0.10, 0.05)
    print("\n=== finding pose above red slot (0.30, -0.10, 0.05) ===")
    target = np.array([0.30, -0.10, 0.05])
    best3 = None
    best_d3 = 1e9
    for shoulder_pan in np.linspace(-1.0, 0.0, 11):
        for shoulder_lift in np.linspace(-1.5, 0.5, 9):
            for elbow_flex in np.linspace(0.5, 1.6, 9):
                for wrist_flex in np.linspace(-1.5, 0.5, 9):
                    q = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, 0.0, 1.6]
                    tip = tip_for_qpos(env, q)
                    if tip[2] < 0.01: continue
                    d = float(np.linalg.norm(tip - target))
                    if d < best_d3:
                        best_d3, best3 = d, (q, tip)
    print(f"  best q for tip ≈ {target} : q={best3[0]}  →  tip={best3[1]}  (d={best_d3:.4f})")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
