"""Offline closed-loop rollout of a deployed policy WITHOUT the real arm.
Assumes perfect servo tracking (qpos == accumulated target), which is what deploy approximates.
Cubes are STATIC (we don't simulate grasping), so this only tests REACH+DESCENT toward the
current cube — does the TCP home onto the cube xy and lower z toward grasp height (~0.01)?

Compares the deploy state the policy actually saw (only red present, 4 distractors at prior)
vs the in-distribution state (all 5 cubes real, as in every demo).

Usage: python offline_rollout.py runs/bc_n5/policy.pt
"""
import sys
import numpy as np
import torch
import deploy_real as D

np.set_printoptions(precision=3, suppress=True)
ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/bc_n5/policy.pt"

agent = D.StateAgent()
agent.load_state_dict(torch.load(ckpt, map_location="cpu"))
agent.eval()
fk = D.SO100ForwardKinematics()

# distractor prior (matches CubeTracker.last_positions init)
prior = np.zeros((5, 3), np.float32)
prior[:, 0] = 0.27
prior[:, 1] = np.linspace(-0.12, 0.12, 5)
prior[:, 2] = D.CUBE_TABLE_Z

# realistic 5-cube START layout from demo ep0's first frame
d = np.load("demos_clean.npz")
cubes_5 = d["states"][0, 9:24].reshape(5, 3).copy()

RED = np.array([0.25, -0.07, D.CUBE_TABLE_Z], np.float32)   # the test cube position


def rollout(cubes0, label, ema=0.3, steps=200):
    cubes = cubes0.copy()
    target = D.REST_QPOS.copy().astype(np.float32)
    prev_a = np.zeros(6, np.float32)
    print(f"\n=== {label} | red target={cubes[0,:2]} ===")
    min_d = 1e9
    for t in range(steps):
        qpos = target.copy()
        tcp = fk.tcp_pos(qpos)
        cur, done = D.compute_current_color_idx(cubes, num_active=1,
                                                detected=np.ones(5, bool))
        state = D.build_state_vector(qpos, tcp, cubes, cur, done)
        with torch.no_grad():
            a = agent.actor_mean(torch.from_numpy(state).float().unsqueeze(0)).squeeze(0).numpy()
        a = ema * prev_a + (1 - ema) * a
        prev_a = a.copy()
        a[4] = 0.0
        a = np.clip(a, -1, 1)
        target = np.clip(target + a * D.ACTION_RANGE,
                         [-2.5, -2.5, -2.5, -2.5, -2.5, 0.0], [2.5, 2.5, 2.5, 2.5, 2.5, 1.5])
        dxy = np.linalg.norm(tcp[:2] - cubes[0, :2])
        min_d = min(min_d, dxy)
        if t % 20 == 0:
            print(f" t{t:3d} tcp=({tcp[0]:+.2f},{tcp[1]:+.2f},{tcp[2]:+.2f}) "
                  f"dxy_to_red={dxy*100:4.1f}cm  a=[{','.join(f'{x:+.2f}' for x in a)}]")
    print(f" --> closest TCP-to-red xy over rollout: {min_d*100:.1f}cm")


# 1) what deploy actually fed: red real, 4 distractors at prior
c1 = prior.copy(); c1[0] = RED
rollout(c1, "DEPLOY-LIKE (only red real, 4 distractors at prior)")
# 2) in-distribution: all 5 cubes real (demo ep0 start), red at that layout
rollout(cubes_5, "IN-DIST (all 5 cubes real, demo ep0 start layout)")
