"""Hand-scripted grasp/pick-and-place oracle (IK-driven).

Validates MuJoCo physics for the pick-and-place loop. If this works, RL can learn
the task. If not, physics needs tuning before any RL is wasted.
"""
from __future__ import annotations

import sys
import pathlib
import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from env.sort_env import SortBlocksEnv, CUBE_REST_Z, TARGET_POSITIONS  # noqa: E402
sys.path.insert(0, str(HERE))
from ik_helper import ik_tip_target  # noqa: E402

OUT_DIR = ROOT / "outputs" / "grasp_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_frames(frames: list[np.ndarray], path: pathlib.Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(frames)
        cols = min(n, 6)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.3))
        if rows == 1: axes = np.array([axes])
        if cols == 1: axes = axes.reshape(-1, 1)
        for i, f in enumerate(frames):
            ax = axes[i // cols, i % cols]
            ax.imshow(f); ax.set_title(f"t={i}", fontsize=7); ax.axis("off")
        for j in range(len(frames), rows * cols):
            axes[j // cols, j % cols].axis("off")
        fig.tight_layout(); fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"  (frame save failed: {e})")


def _solve_keyframes(env: SortBlocksEnv, cube_xy: tuple[float, float],
                     slot_xy: np.ndarray) -> dict[str, np.ndarray]:
    """Build a dict of (name → q[6]) using IK for each waypoint."""
    q_home = np.array([0.0, -0.8, 1.2, -0.4, 0.0, 1.6])
    cx, cy = cube_xy
    sx, sy = slot_xy
    OPEN, CLOSED = 1.6, -0.1

    # Aim ~1.0 cm BELOW where we want the tip to settle to compensate for arm sag
    # under position-control gravity equilibrium. Empirically the actuator settles
    # ~1 cm above the IK solution at low z.
    targets = {
        "above_cube":   (np.array([cx, cy, 0.08]), OPEN),
        "at_cube":      (np.array([cx, cy, 0.005]), OPEN),  # very low target → settles ~cube center
        "lifted":       (np.array([cx, cy, 0.18]), CLOSED),
        "above_slot":   (np.array([sx, sy, 0.18]), CLOSED),
        "at_slot":      (np.array([sx, sy, 0.02]), CLOSED),
        "retreat":      (np.array([sx, sy, 0.18]), OPEN),
    }
    out = {"home": q_home.copy()}
    q_prev = q_home.copy()
    for name, (tgt, grip) in targets.items():
        q, err = ik_tip_target(env, tgt, q_init=q_prev, max_iters=400, lam=0.05, step_clip=0.3)
        # Verify
        env.data.qpos[env._arm_qpos_adr] = q
        mujoco.mj_forward(env.model, env.data)
        tip = env.data.site_xpos[env._tip_site_id].copy()
        q[5] = grip
        out[name] = q
        print(f"  {name:12s} target={tgt}  →  q={q.round(3).tolist()}  tip={tip.round(4).tolist()}  err={err:.4f}")
        q_prev = q.copy()
    return out


def _interp_action(q_from: np.ndarray, q_to: np.ndarray, alpha: float,
                   ctrl_lo: np.ndarray, ctrl_hi: np.ndarray) -> np.ndarray:
    q_target = (1 - alpha) * q_from + alpha * q_to
    a = 2.0 * (q_target - ctrl_lo) / (ctrl_hi - ctrl_lo) - 1.0
    return np.clip(a.astype(np.float32), -1.0, 1.0)


def _run_oracle(
    cube_xy: tuple[float, float] = (0.18, 0.05),
    record_every: int = 6,
    cube_friction: tuple[float, float, float] | None = None,
    seed: int = 0,
) -> dict:
    env = SortBlocksEnv(num_active_cubes=1, render_images=True, image_size=(280, 280), seed=seed)
    obs, info = env.reset(seed=seed)

    # Override cube spawn
    base = env._cube_jnt_qpos_adr[0]
    env.data.qpos[base + 0] = cube_xy[0]
    env.data.qpos[base + 1] = cube_xy[1]
    env.data.qpos[base + 2] = CUBE_REST_Z
    env.data.qpos[base + 3:base + 7] = [1, 0, 0, 0]
    if cube_friction is not None:
        cube_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_red_geom")
        env.model.geom_friction[cube_geom_id, :] = cube_friction
    mujoco.mj_forward(env.model, env.data)

    cube0_z = float(env.data.xpos[env._cube_body_ids[0], 2])
    print(f"  cube spawn xyz = {env.data.xpos[env._cube_body_ids[0]].round(4).tolist()}")

    # Solve keyframes
    kfs = _solve_keyframes(env, cube_xy, TARGET_POSITIONS[0])

    SEGMENTS = [
        ("home",       "above_cube",   25),
        ("above_cube", "at_cube",      25),
        ("at_cube",    "at_cube",      10),   # hold for closing (q updates gripper)
        ("at_cube",    "lifted",       25),   # but lifted has gripper=CLOSED
        ("lifted",     "above_slot",   25),
        ("above_slot", "at_slot",      20),
        ("at_slot",    "retreat",      15),
    ]

    # Need at_cube_closed intermediate for "close gripper before lift" — patch:
    at_cube_closed = kfs["at_cube"].copy()
    at_cube_closed[5] = -0.1
    kfs["at_cube_closed"] = at_cube_closed

    SEGMENTS = [
        ("home",            "above_cube",       25),
        ("above_cube",      "at_cube",          25),
        ("at_cube",         "at_cube_closed",   15),   # close gripper in place
        ("at_cube_closed",  "lifted",           25),
        ("lifted",          "above_slot",       25),
        ("above_slot",      "at_slot",          20),
        ("at_slot",         "retreat",          15),
    ]

    frames = []
    states_log = []
    t = 0
    for kf_from, kf_to, n in SEGMENTS:
        q_from = kfs[kf_from]
        q_to = kfs[kf_to]
        for k in range(n):
            alpha = (k + 1) / n
            a = _interp_action(q_from, q_to, alpha, env._ctrl_lo, env._ctrl_hi)
            obs, r, term, trunc, info = env.step(a)
            cube_pos = env.data.xpos[env._cube_body_ids[0]].copy()
            states_log.append({
                "t": t, "segment": f"{kf_from}→{kf_to}", "step_in_seg": k,
                "cube_xyz": cube_pos.tolist(),
                "tip_xyz": info["tip_xyz"].tolist(),
                "sort_progress": info["sort_progress"],
            })
            if t % record_every == 0 or k == n - 1:
                frames.append(obs["image_front"])
            t += 1
        last = states_log[-1]
        print(f"  {kf_from:18s} → {kf_to:18s}  end_cube_z={last['cube_xyz'][2]:.4f}  "
              f"tip_z={last['tip_xyz'][2]:.4f}  progress={last['sort_progress']}")

    final_cube = env.data.xpos[env._cube_body_ids[0]].copy()
    max_z = max(s["cube_xyz"][2] for s in states_log)
    print(f"\n  RESULT: max_cube_z={max_z:.4f}  final_cube={final_cube.round(4).tolist()}  "
          f"sort_progress={states_log[-1]['sort_progress']}")
    env.close()
    return {
        "frames": frames, "states": states_log,
        "spawn_z": cube0_z, "max_z": max_z,
        "final_xyz": final_cube.tolist(),
        "sort_progress": states_log[-1]["sort_progress"],
    }


def main() -> int:
    print("=== grasp oracle test (IK-driven, default friction) ===")
    res = _run_oracle()
    _save_frames(res["frames"], OUT_DIR / "ik_default.png")

    grasp_ok = res["max_z"] - res["spawn_z"] > 0.03
    place_ok = res["sort_progress"] == 1
    print(f"\n  grasp_ok: {grasp_ok}   place_ok: {place_ok}")
    if grasp_ok and place_ok:
        print("\n=== PASS ===")
        return 0

    print("\n  Trying friction sweep + jaw-grip param boost...")
    sweeps = [
        ("high_fric", (3.0, 0.5, 0.01)),
        ("very_high", (5.0, 1.0, 0.05)),
    ]
    best = res; best_label = "default"
    for label, fric in sweeps:
        print(f"\n--- friction {label} = {fric} ---")
        r = _run_oracle(cube_friction=fric)
        if r["max_z"] - r["spawn_z"] > best["max_z"] - best["spawn_z"]:
            best, best_label = r, label
        _save_frames(r["frames"], OUT_DIR / f"ik_{label}.png")
    print(f"\n  best: {best_label}  lift={best['max_z']-best['spawn_z']:+.4f}")
    return 0 if (best["max_z"] - best["spawn_z"] > 0.03) else 2


if __name__ == "__main__":
    raise SystemExit(main())
