"""Simple numerical IK for SO-ARM101 tip site → target xyz.

Levenberg–Marquardt style: q ← q + (J^T J + λI)^-1 J^T (target - tip).

Used by the grasp oracle to compute joint targets without hand-tuning keyframes.
"""
from __future__ import annotations

import numpy as np
import mujoco


def ik_tip_target(
    env,  # SortBlocksEnv
    target_xyz: np.ndarray,
    q_init: np.ndarray | None = None,
    *,
    max_iters: int = 200,
    tol: float = 1e-3,
    lam: float = 0.1,
    step_clip: float = 0.2,
    keep_gripper: bool = True,
) -> tuple[np.ndarray, float]:
    """Solve for arm qpos s.t. tip site lands near target_xyz.

    Returns (q_solution[6], final_err). Does NOT modify env.data permanently —
    restores arm qpos to its original state on return.
    """
    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    q0 = env.data.qpos[env._arm_qpos_adr].copy()
    if q_init is None:
        q = q0.copy()
    else:
        q = np.asarray(q_init, dtype=np.float64).copy()

    # joint limits from the model
    lo = env.model.jnt_range[env._arm_joint_ids, 0].copy()
    hi = env.model.jnt_range[env._arm_joint_ids, 1].copy()

    # workspace: only the 5 arm joints (skip gripper which doesn't affect tip much)
    n_dof = 5

    err = np.inf
    for _it in range(max_iters):
        # Set qpos and forward
        env.data.qpos[env._arm_qpos_adr] = q
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        tip = env.data.site_xpos[env._tip_site_id].copy()
        err_vec = target - tip
        err = float(np.linalg.norm(err_vec))
        if err < tol:
            break
        # Jacobian of tip site (3 x nv)
        jacp = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacp, None, env._tip_site_id)
        # Subselect arm DoFs
        J = jacp[:, env._arm_qvel_adr[:n_dof]]
        # LM step
        JTJ = J.T @ J + lam * np.eye(n_dof)
        dq = np.linalg.solve(JTJ, J.T @ err_vec)
        # clip step magnitude
        norm_dq = np.linalg.norm(dq)
        if norm_dq > step_clip:
            dq = dq * (step_clip / norm_dq)
        q[:n_dof] += dq
        # clip to joint limits
        q[:n_dof] = np.clip(q[:n_dof], lo[:n_dof], hi[:n_dof])

    if not keep_gripper:
        q[5] = q0[5]
    # restore original state
    env.data.qpos[env._arm_qpos_adr] = q0
    mujoco.mj_forward(env.model, env.data)
    return q, err


if __name__ == "__main__":
    # smoke test
    import sys, pathlib
    HERE = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE.parent))
    from env.sort_env import SortBlocksEnv

    env = SortBlocksEnv(num_active_cubes=1, render_images=False, seed=0)
    env.reset(seed=0)
    targets = [
        ("above cube (0.18, 0.05, 0.05)",  np.array([0.18,  0.05, 0.05])),
        ("at cube   (0.18, 0.05, 0.018)",  np.array([0.18,  0.05, 0.018])),
        ("above red slot (0.30, -0.10, 0.10)", np.array([0.30, -0.10, 0.10])),
        ("at red slot    (0.30, -0.10, 0.04)", np.array([0.30, -0.10, 0.04])),
        ("above blue slot (0.30, +0.10, 0.10)", np.array([0.30,  0.10, 0.10])),
    ]
    for name, tgt in targets:
        q, err = ik_tip_target(env, tgt, q_init=np.array([0.0, -0.8, 1.2, -0.4, 0.0, 1.6]))
        # verify
        env.data.qpos[env._arm_qpos_adr] = q
        mujoco.mj_forward(env.model, env.data)
        tip = env.data.site_xpos[env._tip_site_id].copy()
        print(f"  {name:42s} q={q.round(3).tolist()}  tip={tip.round(4).tolist()}  err={err:.4f}")
    env.close()
