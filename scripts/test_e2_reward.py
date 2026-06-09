"""E2 reward + curriculum sanity test.

Three checks:
  1. Curriculum: instantiate env with num_active_cubes ∈ {1, 3, 5}. Confirm
     active_mask, current_color_idx, and parked-cube positions are correct.
  2. Teleport teleport: in a num_active=1 env, warp red cube into the red slot,
     step once, confirm placement_added=1 + done bonus fired.
  3. Frame-rate: compare render_images=True vs render_images=False step latency.

Run:
    $env:PYTHONPATH = "D:\\soarm101_sorting"
    & "C:\\Users\\asus\\miniconda3\\envs\\lerobot\\python.exe" `
      "D:\\soarm101_sorting\\scripts\\test_e2_reward.py"
"""
from __future__ import annotations

import sys
import time
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from env.sort_env import (  # noqa: E402
    SortBlocksEnv, CUBE_COLORS, TARGET_POSITIONS, CUBE_REST_Z,
    PARK_X, HOME_QPOS_ARM,
)


def _check_curriculum(n: int) -> None:
    print(f"\n--- curriculum n={n} ---")
    env = SortBlocksEnv(num_active_cubes=n, render_images=False, seed=7)
    obs, info = env.reset(seed=7)

    print(f"  active_mask        = {info['active_mask'].astype(int).tolist()}")
    print(f"  placed_mask        = {info['placed_mask'].astype(int).tolist()}")
    print(f"  current_color_idx  = {info['current_color_idx']} "
          f"({CUBE_COLORS[info['current_color_idx']] if info['current_color_idx']>=0 else 'DONE'})")
    print(f"  obs['state'].shape = {obs['state'].shape}")
    # active_mask + current_onehot are the last 10 elements of state
    print(f"  state[15:20] (active_mask)        = {obs['state'][15:20].astype(int).tolist()}")
    print(f"  state[20:25] (current_onehot)     = {obs['state'][20:25].astype(int).tolist()}")
    print(f"  cube xy positions (active first):")
    for i, c in enumerate(CUBE_COLORS):
        x, y, z = obs['cube_positions'][i]
        tag = "ACTIVE" if info['active_mask'][i] else "PARKED"
        print(f"    {c:7s} [{tag}] xyz = ({x:+.3f}, {y:+.3f}, {z:+.3f})")
    # Sanity check parking
    parked = ~info['active_mask']
    if parked.any():
        x_parked = obs['cube_positions'][parked, 0]
        assert (x_parked >= PARK_X - 0.05).all(), \
            f"parked cubes should be at x≥{PARK_X}, got x={x_parked}"
        print(f"  OK: all parked cubes at x≥{PARK_X-0.05:.2f} (out of reach + out of FOV)")
    env.close()


def _check_placement_teleport() -> None:
    print("\n--- placement teleport test (num_active=1, warp red cube to slot) ---")
    env = SortBlocksEnv(num_active_cubes=1, render_images=False, seed=1)
    obs, info = env.reset(seed=1)
    print(f"  initial sort_progress = {info['sort_progress']} / 1")

    # Manually warp red cube (idx 0) into red slot. Free-joint qpos = [x, y, z, qw, qx, qy, qz]
    base = env._cube_jnt_qpos_adr[0]
    env.data.qpos[base + 0] = TARGET_POSITIONS[0, 0]
    env.data.qpos[base + 1] = TARGET_POSITIONS[0, 1]
    env.data.qpos[base + 2] = CUBE_REST_Z
    env.data.qpos[base + 3:base + 7] = [1, 0, 0, 0]
    # Zero cube velocity too so it doesn't immediately bounce out
    qveladr = env.model.jnt_dofadr[
        env.model.body_jntadr[env._cube_body_ids[0]]
    ]
    env.data.qvel[qveladr:qveladr + 6] = 0.0
    import mujoco
    mujoco.mj_forward(env.model, env.data)

    # Step with "hold home" action (mapped from current home qpos)
    home_action = 2.0 * (HOME_QPOS_ARM - env._ctrl_lo) / (env._ctrl_hi - env._ctrl_lo) - 1.0
    obs, r, term, trunc, info = env.step(home_action.astype(np.float32))
    comp = info["reward_components"]
    print(f"  reward = {r:+.3f}")
    print(f"  components: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in comp.items()))
    print(f"  placed_mask = {info['placed_mask'].astype(int).tolist()}, "
          f"sort_progress = {info['sort_progress']} / 1")
    print(f"  terminated = {term}, is_success = {info['is_success']}")
    # Strict assertions
    assert comp["placement_added"] == 1, f"expected placement_added=1, got {comp['placement_added']}"
    assert comp["done"] == 10.0, f"expected done bonus, got {comp['done']}"
    assert info["is_success"], "should be success"
    assert term, "should be terminated"
    print("  OK: placement bonus + done bonus fired, episode terminated.")
    env.close()


def _check_latency() -> None:
    print("\n--- step latency: render_images=False vs True ---")
    rng = np.random.default_rng(0)
    for render in (False, True):
        env = SortBlocksEnv(num_active_cubes=5, render_images=render, seed=0)
        env.reset(seed=0)
        # Warm-up step
        env.step(np.zeros(6, dtype=np.float32))
        n = 100
        t0 = time.perf_counter()
        for _ in range(n):
            a = rng.uniform(-0.2, 0.2, size=6).astype(np.float32)
            env.step(a)
        dt = time.perf_counter() - t0
        print(f"  render_images={render!s:5s}  -> {1000*dt/n:6.2f} ms/step  ({n/dt:6.1f} steps/s)")
        env.close()


def main() -> int:
    print("=== E2 reward + curriculum sanity ===")
    for n in (1, 3, 5):
        _check_curriculum(n)
    _check_placement_teleport()
    _check_latency()
    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
