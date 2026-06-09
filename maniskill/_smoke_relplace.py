"""Smoke + logic test for relative_placement mode.
(1) unit-test the sorted-row criterion on hand-crafted cube layouts (pure torch, no sim).
(2) build the real env with relative_placement=True, step it, confirm no crash + sane shapes.
"""
import os
os.environ.setdefault("SORT_CUBES_NUM_ACTIVE", "5")
import numpy as np
import torch
import sort_cubes_env as E


def rel_placed(cy, cx, cz):
    """Replicate evaluate()'s relative criterion for one env. cy/cx/cz: (5,) np arrays."""
    on_table = cz < E.PLACEMENT_Z_MAX
    in_band = (cx > E.REL_PLACE_X_MIN) & (cx < E.REL_PLACE_X_MAX)
    placed = np.zeros(5, bool)
    prev_p, prev_y = True, np.inf
    for i in range(5):
        base = on_table[i] and in_band[i]
        if i == 0:
            placed[i] = base and (cy[0] >= E.REL_RED_Y_MIN)
        else:
            placed[i] = base and prev_p and (cy[i] <= prev_y - E.REL_MIN_GAP)
        prev_p, prev_y = placed[i], cy[i]
    return placed


print("=== unit test: relative sorted-row criterion ===")
X = np.full(5, 0.32)
Z = np.full(5, 0.013)
cases = {
    "perfect sorted row (demo layout)": np.array([0.09, 0.03, -0.03, -0.09, -0.14]),
    "floating row (shifted right, still sorted)": np.array([0.05, 0.00, -0.05, -0.10, -0.15]),
    "red too far right (no anchor)": np.array([-0.10, -0.12, -0.14, -0.16, -0.18]),
    "two cubes swapped (O left of R-region)": np.array([0.03, 0.09, -0.03, -0.09, -0.14]),
    "gap too small between R and O": np.array([0.09, 0.07, -0.03, -0.09, -0.14]),
    "all bunched (no gaps)": np.array([0.02, 0.02, 0.02, 0.02, 0.02]),
}
for name, cy in cases.items():
    p = rel_placed(cy, X, Z)
    print(f"  {name:48s} placed={p.astype(int)}  all={p.all()}")

print("\n=== env smoke: relative_placement=True ===")
import gymnasium as gym
import mani_skill.envs  # noqa
env = gym.make("SortCubesSO100-v1", num_envs=4, obs_mode="state", sim_backend="physx_cuda",
               control_mode="pd_joint_target_delta_pos", num_active_cubes=5, relative_placement=True)
obs, _ = env.reset(seed=0)
print(f"  reset OK, obs {obs.shape}")
for t in range(5):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
print(f"  stepped 5x OK. n_placed={info['n_placed'].cpu().tolist()} "
      f"success={info['success'].cpu().tolist()} reward={r.cpu().numpy().round(2).tolist()}")
print(f"  in_slot[0]={info['in_slot'][0].cpu().int().tolist()}")
env.close()
print("OK")
