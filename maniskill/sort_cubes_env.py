"""SortCubesSO100-v1 — 5 colored cubes sorted into 5 matching slots by SO-100 arm.

Subclasses ManiSkill's SO100GraspCubeEnv. GPU-parallelized via SAPIEN — all reward
and evaluate logic uses torch tensors broadcast over num_envs.

Task (strict-order R→O→Y→G→B):
  - 5 cubes (red, orange, yellow, green, blue) spawn at random positions in workspace
  - 5 target slot xy positions, fixed, one per color
  - "current color" = lowest color index not yet placed
  - Reward (ManiSkill-style, paired with gamma=0.9):
      reach = 1 - tanh(5 × dist(tcp, current_cube))   when not grasping
      grasp = float(is_grasping_current)               per-step while holding
      transport = is_grasping_current × (1 - tanh(5 × dist(current_cube, current_slot)))
      placement = +N_PLACE_BONUS one-shot per cube transitioned into its slot
      done = +N_DONE one-shot when all 5 placed
      table_penalty = -2 × float(touching_table)
  - Success: all 5 cubes within PLACEMENT_XY_TOL of their slots, z below threshold

Note: SAPIEN has full contact physics — magnetic grasp NOT used here. Tests if
SO-100 jaws can actually grasp small cubes in SAPIEN (vs MuJoCo which couldn't).
"""
from __future__ import annotations
import os

# Windows SAPIEN CUDA workaround (BEFORE any sapien import)
if os.path.exists(r"D:\soarm101\cuda_workaround"):
    os.add_dll_directory(r"D:\soarm101\cuda_workaround")
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(_cuda_bin):
    os.add_dll_directory(_cuda_bin)

from typing import Any
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.digital_twins.so100_arm.grasp_cube import (
    SO100GraspCubeEnv,
)
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder


N_CUBES = 5
CUBE_COLORS_RGBA = np.array([
    [0.85, 0.10, 0.10, 1.0],  # red
    [0.95, 0.50, 0.10, 1.0],  # orange
    [0.95, 0.90, 0.10, 1.0],  # yellow
    [0.15, 0.75, 0.20, 1.0],  # green
    [0.10, 0.30, 0.85, 1.0],  # blue
], dtype=np.float32)

# Target slot xy positions in robot frame (workspace is in front of robot at +x).
# Layout adopted from the user's OWN teleop-demo placements (measured realized means): red on
# the LEFT (+y), blue on the RIGHT (-y), at x~0.32. This is the y-MIRROR of the old 0.28 row;
# retraining on it makes the recorded demos self-consistent for BC. MUST match deploy_real.SLOT_XY.
SLOT_XY = np.array([
    [0.32,  0.09],  # red    (front-LEFT)
    [0.32,  0.03],  # orange
    [0.32, -0.03],  # yellow
    [0.32, -0.09],  # green
    [0.32, -0.14],  # blue   (front-RIGHT)
], dtype=np.float32)

# Spawn region — all 5 cubes always spawn in workspace (Option D).
# Wider y-range to fit 5 cubes safely:
#   N_CUBES=5 bands of width (0.30/5)=0.06 each, cube width 0.025,
#   so adjacent cube centers ≥ 0.06 - 2*jitter(0.012) = 0.036 > 0.025 (no collision).
SPAWN_X_RANGE = (0.16, 0.25)   # Path B: pulled in (behind slots at 0.28), all graspable
SPAWN_Y_RANGE = (-0.12, 0.12)  # slightly tighter; cubes still in separate y-bands (no collision)
SPAWN_Z = 0.013      # cube center z when resting on table (cube half_size ≈ 0.0125)
CUBE_HALF_SIZE = 0.0125
MIN_PAIR_DIST = 0.045
# Jitter within each y-band as fraction of band height (0.6 → ±0.018 = ±band/3.3)
Y_BAND_JITTER_FRAC = 0.6

# Placement tolerances
PLACEMENT_XY_TOL = 0.025  # 2.5 cm tolerance (slot markers are 3cm wide)
PLACEMENT_Z_MAX  = 0.05   # cube must be near table to count as placed

# Relative-placement mode parameters (used iff relative_placement=True). The placed cubes must
# form a left->right sorted row inside a reachable front band; positions float (no fixed slot).
REL_PLACE_X_MIN  = 0.27   # cube x must be in this front band to count as "placed in the row"
REL_PLACE_X_MAX  = 0.37
REL_RED_Y_MIN    = 0.02   # the FIRST cube (red) must land left-of-center (y >= this) so 4 fit to its right
REL_MIN_GAP      = 0.04   # each next cube must be >= this much to the RIGHT (smaller y) than the previous

# Reward weights — Path B: CONTINUOUS staged design (no large discrete one-shots).
# Per current cube: staged = reach[0..1] + is_grasp[0/1] + grasp-gated place[0..1] (max 3).
# base = KP_PROGRESS * num_placed, given EVERY step (monotone progress, never re-farmed).
# KP_PROGRESS MUST exceed STAGED_MAX(=3): then lowering-to-place + advancing to next cube
# strictly beats hovering-grasped-at-slot-xy (the classic farming exploit). Verified by
# the gamma-bounded farming check before training.
KP_PROGRESS   = 6.0   # per-step reward per cube already placed (MUST be > staged_max=4)
W_GRASP_STAGE = 2.0   # weight on is_grasp in staged: makes grasping clearly beat hovering
                      # (incentive ladder hover<grasp<place) -> aids cold-start grasp discovery
STAGED_MAX    = 4.0   # reach(1) + W_GRASP_STAGE*grasp(2) + place(1)
W_SUCCESS     = 10.0  # per-step bonus while all active cubes placed (small terminal)
W_TABLE_PEN   = -2.0  # per step touching table (safety)
W_SMOOTH      = 0.1   # action-rate penalty ||a_t - a_{t-1}|| (sim2real: damp oscillation)
REACH_TANH_K  = 5.0
PLACE_TANH_K  = 5.0


@register_env("SortCubesSO100-v1", max_episode_steps=350)
class SortCubesSO100Env(SO100GraspCubeEnv):
    """5-cube ordered sort task for SO-100 arm. Reuses parent's robot + table +
    camera + lighting + greenscreen setup; replaces cube building + task logic."""

    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb+segmentation"]

    def __init__(self, *args, num_active_cubes: int = N_CUBES,
                 randomize_active_color: bool = False,
                 flexible_order: bool = False,
                 prefill_min: int = 0, prefill_max: int = 0,
                 fix_wrist_roll: bool = True,
                 sparse_reward: bool = False,
                 relative_placement: bool = False,
                 # Domain Randomization (Phase B). All off by default for backward compat.
                 dr_physics: bool = False,
                 dr_obs_noise_std: float = 0.0,   # meters, additive to position obs
                 dr_action_noise_std: float = 0.0, # additive to action in [-1,1] range
                 dr_cube_size_jitter: float = 0.0, # ± fraction of cube half_size
                 **kwargs):
        if not (1 <= num_active_cubes <= N_CUBES):
            raise ValueError(f"num_active_cubes must be in [1, {N_CUBES}]")
        self.num_active_cubes = int(num_active_cubes)
        self.randomize_active_color = bool(randomize_active_color)
        self.flexible_order = bool(flexible_order)
        self.prefill_min = int(prefill_min)
        self.prefill_max = int(prefill_max)
        if self.prefill_max > self.num_active_cubes - 1:
            raise ValueError(f"prefill_max must be < num_active_cubes")
        if self.prefill_min > self.prefill_max:
            raise ValueError("prefill_min > prefill_max")
        # Fix joint 4 (wrist roll) — doesn't affect top-down pick-and-place since
        # cubes and slots are rotationally symmetric. Eval trace showed actor_logstd[4]
        # =1.67 (almost random) confirming policy learned it's irrelevant. Fixing it
        # reduces effective action space 6D → 5D, less exploration noise, faster
        # convergence, more stable sim2real (real wrist won't drift randomly).
        self.fix_wrist_roll = bool(fix_wrist_roll)
        # Sparse reward mode: reward = # active cubes currently in slot (0..num_active),
        # no shaping, no negatives. Use ONLY with a warm-start (else never sees success).
        self.sparse_reward = bool(sparse_reward)
        # Relative-placement mode: instead of fixed per-cube slots, success only requires the
        # cubes to form a left->right sorted ROW (red leftmost) within a reachable front band —
        # each next cube placed to the RIGHT of the previous with a min gap. Targets float; the
        # nominal SLOT_XY row is kept only as an obs hint + reward-shaping anchor (so demos and
        # checkpoints stay obs-compatible). See evaluate().
        self.relative_placement = bool(relative_placement)
        # DR flags
        self.dr_physics = bool(dr_physics)
        self.dr_obs_noise_std = float(dr_obs_noise_std)
        self.dr_action_noise_std = float(dr_action_noise_std)
        self.dr_cube_size_jitter = float(dr_cube_size_jitter)
        if not (0.0 <= self.dr_cube_size_jitter <= 0.5):
            raise ValueError("dr_cube_size_jitter must be in [0, 0.5]")
        super().__init__(*args, **kwargs)

    # Override step to (1) clamp action[4]=0 if fix_wrist_roll, (2) inject action noise.
    # Keeps action_space 6D for ckpt compatibility but makes joint 4 a no-op when fixed.
    def step(self, action):
        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.clone()
                # Inject DR action noise BEFORE clamping joint 4 (so noise on joint 4 also zeroed)
                if self.dr_action_noise_std > 0:
                    noise = torch.randn_like(action) * self.dr_action_noise_std
                    action = action + noise
                if self.fix_wrist_roll:
                    action[..., 4] = 0.0
            else:
                action = action.copy()
                if self.dr_action_noise_std > 0:
                    action = action + np.random.randn(*action.shape).astype(action.dtype) * self.dr_action_noise_std
                if self.fix_wrist_roll:
                    action[..., 4] = 0.0
        return super().step(action)

    # ============================================================
    # Scene loading: 5 cubes + 5 visual slot markers
    # ============================================================
    def _load_scene(self, options: dict):
        # 1. Table (reuse parent's TableSceneBuilder)
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # 2. Build N_CUBES merged actors (one per color)
        # DR: optionally randomize per-env friction/density/size at scene build time.
        # Note: SAPIEN materials are fixed at build; per-episode physics changes
        # would require reconfigure (expensive). Per-env is sufficient for DR.
        # We pre-sample so cubes within same env share params (consistent).
        rng = np.random.default_rng(seed=0)  # deterministic per scene build
        self.cubes = []
        # Per-env DR params (also stored for inspection)
        self._cube_half_size_per_env = np.full((self.num_envs,), CUBE_HALF_SIZE, dtype=np.float32)
        for ci in range(N_CUBES):
            cubes_for_color = []
            for env_i in range(self.num_envs):
                # Sample per-env DR params (same for all cubes within one env)
                if self.dr_physics:
                    # rng sequence depends only on env_i for consistency across colors
                    env_rng = np.random.default_rng(seed=1000 + env_i)
                    friction = float(env_rng.uniform(0.3, 0.8))
                    density = float(env_rng.uniform(160, 240))  # default 200, ±20%
                else:
                    friction = 0.5
                    density = 200.0
                if self.dr_cube_size_jitter > 0:
                    env_rng_sz = np.random.default_rng(seed=2000 + env_i)
                    sz_mult = float(env_rng_sz.uniform(
                        1.0 - self.dr_cube_size_jitter, 1.0 + self.dr_cube_size_jitter))
                    half_size = CUBE_HALF_SIZE * sz_mult
                    self._cube_half_size_per_env[env_i] = half_size
                else:
                    half_size = CUBE_HALF_SIZE
                builder = self.scene.create_actor_builder()
                material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=friction, dynamic_friction=friction, restitution=0.0,
                )
                builder.add_box_collision(
                    half_size=[half_size] * 3, material=material, density=density,
                )
                builder.add_box_visual(
                    half_size=[half_size] * 3,
                    material=sapien.render.RenderMaterial(
                        base_color=CUBE_COLORS_RGBA[ci].tolist(),
                    ),
                )
                # Initial pose far enough apart so no auto-collision during build
                builder.initial_pose = sapien.Pose(p=[0, 0.5 + 0.1 * ci, half_size])
                builder.set_scene_idxs([env_i])
                actor = builder.build(name=f"cube_{ci}_env{env_i}")
                cubes_for_color.append(actor)
                self.remove_from_state_dict_registry(actor)
            merged = Actor.merge(cubes_for_color, name=f"cube_{ci}")
            self.add_to_state_dict_registry(merged)
            self.cubes.append(merged)

        # 3. Slot visual markers (kinematic, no collision, low-profile cylinders)
        # We use small thin boxes as markers. Per-env merged actors.
        self.slot_markers = []
        for si in range(N_CUBES):
            markers_for_slot = []
            for env_i in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                builder.add_box_visual(
                    half_size=[0.015, 0.015, 0.0005],  # 3cm × 3cm × 1mm flat marker
                    material=sapien.render.RenderMaterial(
                        # 50% alpha for slot markers
                        base_color=[CUBE_COLORS_RGBA[si, 0], CUBE_COLORS_RGBA[si, 1],
                                    CUBE_COLORS_RGBA[si, 2], 0.5],
                    ),
                )
                builder.initial_pose = sapien.Pose(
                    p=[float(SLOT_XY[si, 0]), float(SLOT_XY[si, 1]), 0.0005],
                )
                builder.set_scene_idxs([env_i])
                marker = builder.build_kinematic(name=f"slot_{si}_env{env_i}")
                markers_for_slot.append(marker)
                self.remove_from_state_dict_registry(marker)
            merged = Actor.merge(markers_for_slot, name=f"slot_{si}")
            self.slot_markers.append(merged)

        # 4. Greenscreen + robot color + camera_mount (reuse parent's pattern)
        # We don't want the slot markers greenscreened (they're task-relevant visual cues)
        self.remove_object_from_greenscreen(self.agent.robot)
        for c in self.cubes:
            self.remove_object_from_greenscreen(c)
        for m in self.slot_markers:
            self.remove_object_from_greenscreen(m)

        # Hardcoded rest qpos + table pose (from parent)
        self.rest_qpos = torch.tensor(
            [0, 0, 0, np.pi / 2, np.pi / 2, 0],
            device=self.device,
        )
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2),
        )

        # Camera mount
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("camera_mount")

        # Per-cube fixed half-size (no DR here for simplicity in v1)
        self.cube_half_sizes = torch.full(
            (self.num_envs,), CUBE_HALF_SIZE, device=self.device,
        )

        # 5. Per-episode bookkeeping tensors (reset in _initialize_episode)
        self.ever_placed = torch.zeros(
            self.num_envs, N_CUBES, dtype=torch.bool, device=self.device,
        )
        self.done_fired = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self.grasp_oneshot_fired = torch.zeros(
            self.num_envs, N_CUBES, dtype=torch.bool, device=self.device,
        )
        self.reach_oneshot_fired = torch.zeros(
            self.num_envs, N_CUBES, dtype=torch.bool, device=self.device,
        )
        self.transport_oneshot_fired = torch.zeros(
            self.num_envs, N_CUBES, dtype=torch.bool, device=self.device,
        )

        # Per-env active mask: (num_envs, N_CUBES). When randomize_active_color=False,
        # always first num_active_cubes are True (fixed order R→...). When True, each
        # episode randomly picks a subset of size num_active_cubes — current_color_idx
        # still follows lowest-index-first, so policy learns to follow the onehot.
        self.active_mask = torch.zeros(self.num_envs, N_CUBES, dtype=torch.bool, device=self.device)
        if not self.randomize_active_color:
            self.active_mask[:, : self.num_active_cubes] = True

        # Slot positions as torch tensor (cached on device)
        self.slot_xy_t = torch.tensor(SLOT_XY, dtype=torch.float32, device=self.device)

    # ============================================================
    # Episode initialization: spawn cubes + reset bookkeeping
    # ============================================================
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # Robot to rest qpos + slight noise
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.02
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )

            # Option D: ALWAYS spawn all 5 cubes in workspace (no parking).
            # active_mask only controls success/reward — inactive cubes are
            # visible distractors that policy must learn to ignore. This eliminates
            # the curriculum-stage distribution shift that broke n=k → n=k+1 transfer.
            spawn_x_lo, spawn_x_hi = SPAWN_X_RANGE
            spawn_y_lo, spawn_y_hi = SPAWN_Y_RANGE
            y_band_height = (spawn_y_hi - spawn_y_lo) / N_CUBES

            # Prefill curriculum: per env decide which cubes (subset of active) start
            # already at their correct slot. prefill_count ∈ [prefill_min, prefill_max].
            # We choose the LOWEST-INDEX active cubes as the prefilled ones, so the
            # policy still faces "place red→orange→...→blue" in the active prefix order.
            prefill_mask = torch.zeros(b, N_CUBES, dtype=torch.bool, device=self.device)
            if self.prefill_max > 0:
                # Per-env random count in [prefill_min, prefill_max]
                prefill_counts = torch.randint(
                    self.prefill_min, self.prefill_max + 1, (b,), device=self.device,
                )
                # Use the per-env active mask (may have just been re-sampled below; but for
                # default randomize_active_color=False it's set from __init__)
                # Apply: first `count` active cubes of each env are prefilled
                active_per_env = self.active_mask[env_idx]  # (b, N_CUBES)
                # cumulative active count along cube dim
                cum_active = active_per_env.long().cumsum(dim=-1)  # (b, N_CUBES)
                # Cube is prefilled if it's active AND its cum_active index <= prefill_count
                prefill_mask = active_per_env & (cum_active <= prefill_counts.unsqueeze(-1))

            for ci in range(N_CUBES):
                # For envs where cube ci is prefilled → spawn at slot ci
                # For envs where not prefilled → spawn in workspace
                xs = spawn_x_lo + torch.rand(b, device=self.device) * (spawn_x_hi - spawn_x_lo)
                y_band_center = spawn_y_lo + (ci + 0.5) * y_band_height
                ys = y_band_center + (torch.rand(b, device=self.device) - 0.5) * y_band_height * Y_BAND_JITTER_FRAC
                zs = torch.full((b,), SPAWN_Z, device=self.device)
                # Override for prefilled envs: place at slot xy with small noise
                pf = prefill_mask[:, ci]
                if pf.any():
                    slot_x = float(SLOT_XY[ci, 0])
                    slot_y = float(SLOT_XY[ci, 1])
                    noise = (torch.rand(b, 2, device=self.device) - 0.5) * 0.01  # ±5mm noise
                    xs = torch.where(pf, slot_x + noise[:, 0], xs)
                    ys = torch.where(pf, slot_y + noise[:, 1], ys)
                    # z stays SPAWN_Z = 0.013 — on table, will be detected as placed
                xyz = torch.stack([xs, ys, zs], dim=-1)
                yaw = (torch.rand(b, device=self.device) * 2 - 1) * np.pi
                qw = torch.cos(yaw / 2); qz = torch.sin(yaw / 2)
                quat = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)
                self.cubes[ci].set_pose(Pose.create_from_pq(xyz, quat))

            # Reset bookkeeping for these envs
            self.ever_placed[env_idx] = False
            self.done_fired[env_idx] = False
            self.grasp_oneshot_fired[env_idx] = False
            self.reach_oneshot_fired[env_idx] = False
            self.transport_oneshot_fired[env_idx] = False
            if hasattr(self, "prev_action"):
                self.prev_action[env_idx] = 0.0  # reset action-smoothness memory

            # Mark prefilled cubes as already-placed in bookkeeping
            # so place_bonus / transport_oneshot don't fire for them (no free reward).
            # prefill_mask was computed above (False everywhere if prefill_max=0).
            if self.prefill_max > 0:
                # Build full-size masks indexed by env_idx
                # Note: env_idx indexes into full (num_envs, N_CUBES) tensors
                self.ever_placed[env_idx] = prefill_mask
                self.transport_oneshot_fired[env_idx] = prefill_mask
                self.grasp_oneshot_fired[env_idx] = prefill_mask
                self.reach_oneshot_fired[env_idx] = prefill_mask

            # Re-sample active mask if randomize_active_color (independent per env)
            if self.randomize_active_color:
                # For each env in env_idx, pick a random size-num_active_cubes subset of {0..N_CUBES-1}
                new_masks = torch.zeros(b, N_CUBES, dtype=torch.bool, device=self.device)
                for i in range(b):
                    perm = torch.randperm(N_CUBES, device=self.device)
                    chosen = perm[: self.num_active_cubes]
                    new_masks[i, chosen] = True
                self.active_mask[env_idx] = new_masks

            # Re-place slot markers (in case set_pose drift)
            for si in range(N_CUBES):
                marker_xyz = torch.tensor(
                    [SLOT_XY[si, 0], SLOT_XY[si, 1], 0.0005],
                    device=self.device,
                ).unsqueeze(0).expand(b, 3)
                quat0 = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self.device,
                ).unsqueeze(0).expand(b, 4)
                self.slot_markers[si].set_pose(Pose.create_from_pq(marker_xyz, quat0))

    # ============================================================
    # Override agent obs to be stable across ManiSkill versions / configs:
    # only return qpos+qvel, exclude controller.target_qpos which varies.
    # ============================================================
    def _get_obs_agent(self):
        # Only qpos (6 dim) — matches the obs dim trained ckpts expect (42 = 6 + 36 extras)
        # Excludes qvel (was inadvertently present in some training runs) and target_qpos.
        return dict(qpos=self.agent.robot.get_qpos())

    # ============================================================
    # Observation extras: cube positions + current color one-hot + slot positions
    # ============================================================
    def _get_obs_extra(self, info: dict):
        # Stack cube positions: (num_envs, N_CUBES, 3)
        cube_positions = torch.stack(
            [c.pose.p for c in self.cubes], dim=1,
        )
        # Distance to current cube as a privileged signal (helps state policy)
        # current_color_idx broadcast
        current_color_idx = info["current_color_idx"]
        batch = torch.arange(self.num_envs, device=self.device)
        current_cube_xyz = cube_positions[batch, current_color_idx]

        # One-hot encoding of current color (handle "all done" case → all zeros)
        all_done = info.get("success", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        onehot = torch.zeros(self.num_envs, N_CUBES, device=self.device)
        onehot[batch, current_color_idx] = 1.0
        onehot = onehot * (~all_done).float().unsqueeze(-1)  # zero out when done

        # Compute base obs (privileged, no noise)
        tcp_pos = self.agent.tcp_pos
        tcp_to_obj = current_cube_xyz - tcp_pos
        cube_pos_flat = cube_positions.reshape(self.num_envs, -1)

        # DR: inject Gaussian noise on position observations to simulate
        # real-robot sensor noise (camera-based pose estimation, joint encoders).
        # NOT applied to: current_color_onehot (categorical), slot_xy (constants).
        if self.dr_obs_noise_std > 0:
            s = self.dr_obs_noise_std
            tcp_pos = tcp_pos + torch.randn_like(tcp_pos) * s
            tcp_to_obj = tcp_to_obj + torch.randn_like(tcp_to_obj) * s
            cube_pos_flat = cube_pos_flat + torch.randn_like(cube_pos_flat) * s

        obs = dict(
            tcp_to_obj_pos=tcp_to_obj,
            cube_positions=cube_pos_flat,
            current_color_onehot=onehot,
            slot_xy=self.slot_xy_t.reshape(-1).unsqueeze(0).expand(self.num_envs, -1),
            tcp_pos=tcp_pos,
        )
        return obs

    # ============================================================
    # Evaluation: per-cube placement, current color, success
    # ============================================================
    def evaluate(self):
        # cube positions: (num_envs, N_CUBES, 3)
        cube_positions = torch.stack(
            [c.pose.p for c in self.cubes], dim=1,
        )
        cubes_xy = cube_positions[..., :2]   # (num_envs, N_CUBES, 2)
        cubes_z = cube_positions[..., 2]     # (num_envs, N_CUBES)

        # Distance each cube to its slot
        slots_xy_b = self.slot_xy_t.unsqueeze(0)  # (1, N_CUBES, 2)
        dist_cube_to_slot = torch.norm(cubes_xy - slots_xy_b, dim=-1)  # (num_envs, N_CUBES)

        # In-slot: how a cube counts as "placed".
        if self.relative_placement:
            # Relative sorted-row criterion (no fixed slots): cube on table, inside the front
            # x-band, and to the RIGHT of the previous cube by >= REL_MIN_GAP. Red anchors left.
            on_table = cubes_z < PLACEMENT_Z_MAX
            in_band_x = (cubes_xy[..., 0] > REL_PLACE_X_MIN) & (cubes_xy[..., 0] < REL_PLACE_X_MAX)
            cy = cubes_xy[..., 1]  # (num_envs, N_CUBES)
            placed_cols = []
            prev_placed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            prev_y = torch.full((self.num_envs,), float("inf"), device=self.device)
            for i in range(N_CUBES):
                base_ok = on_table[:, i] & in_band_x[:, i]
                if i == 0:
                    placed_i = base_ok & (cy[:, 0] >= REL_RED_Y_MIN)
                else:
                    placed_i = base_ok & prev_placed & (cy[:, i] <= prev_y - REL_MIN_GAP)
                placed_cols.append(placed_i)
                prev_placed = placed_i
                prev_y = cy[:, i]  # reference next cube against this cube's ACTUAL y
            in_slot = torch.stack(placed_cols, dim=1)  # (num_envs, N_CUBES)
        else:
            # Fixed-slot criterion: within tol of the cube's own slot AND released to table.
            in_slot = (dist_cube_to_slot < PLACEMENT_XY_TOL) & (cubes_z < PLACEMENT_Z_MAX)
        # Mask inactive cubes to "placed" (so curriculum works: success iff all active are placed)
        in_slot_for_success = in_slot | (~self.active_mask)

        # "Current" cube selection
        unfilled = self.active_mask & ~in_slot   # (num_envs, N_CUBES)
        if self.flexible_order:
            # Pick NEAREST unfilled active cube — policy chooses dynamic order.
            tcp_pos = self.agent.tcp_pos  # (num_envs, 3)
            tcp_to_cubes = torch.norm(
                cube_positions - tcp_pos.unsqueeze(1), dim=-1
            )  # (num_envs, N_CUBES)
            # Mask out non-unfilled cubes by setting their dist to +inf
            masked_dist = tcp_to_cubes.masked_fill(~unfilled, float("inf"))
            current_color_idx = masked_dist.argmin(dim=-1)  # (num_envs,)
            # If all filled, argmin returns 0; that's fine (success handles)
        else:
            # Strict order: lowest-index unfilled active cube
            current_color_idx = unfilled.int().argmax(dim=-1)  # (num_envs,)

        # success: all active cubes placed
        success = in_slot_for_success.all(dim=-1)

        # tcp_to_current_cube_dist (used in reward)
        batch = torch.arange(self.num_envs, device=self.device)
        current_cube_xyz = cube_positions[batch, current_color_idx]
        tcp_to_current_dist = torch.norm(
            current_cube_xyz - self.agent.tcp_pos, dim=-1,
        )

        # is_grasping_current: check if agent is grasping the current_color cube
        is_grasping_per_cube = torch.stack(
            [self.agent.is_grasping(c) for c in self.cubes], dim=-1,
        )  # (num_envs, N_CUBES)
        is_grasping_current = is_grasping_per_cube[batch, current_color_idx]

        # touching_table (safety / penalty)
        l_contact = self.scene.get_pairwise_contact_forces(
            self.agent.finger1_link, self.table_scene.table,
        )
        r_contact = self.scene.get_pairwise_contact_forces(
            self.agent.finger2_link, self.table_scene.table,
        )
        touching_table = (torch.norm(l_contact, dim=-1) > 1e-2) | (
            torch.norm(r_contact, dim=-1) > 1e-2
        )

        n_placed = (in_slot & self.active_mask).sum(dim=-1)
        n_active = self.active_mask.sum(dim=-1).clamp(min=1)  # avoid div-by-zero
        # Partial-progress metrics (logged by ManiSkill as eval_<name>_once_mean / at_end_mean)
        success_1plus = n_placed >= 1
        success_2plus = n_placed >= 2
        success_3plus = n_placed >= 3
        success_4plus = n_placed >= 4
        return {
            "success": success,
            "success_1plus": success_1plus,
            "success_2plus": success_2plus,
            "success_3plus": success_3plus,
            "success_4plus": success_4plus,
            "n_placed": n_placed,
            "frac_placed": n_placed.float() / n_active.float(),
            "current_color_idx": current_color_idx,
            "in_slot": in_slot,
            "dist_cube_to_slot": dist_cube_to_slot,
            "is_grasping_per_cube": is_grasping_per_cube,
            "is_grasping_current": is_grasping_current,
            "tcp_to_current_dist": tcp_to_current_dist,
            "touching_table": touching_table,
        }

    # ============================================================
    # Dense reward
    # ============================================================
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Path B continuous staged reward (no large discrete one-shots).

        reward = KP_PROGRESS * num_placed         # monotone progress (per step)
               + reach + is_grasp + place         # staged shaping for CURRENT cube (0..3)
               + W_SUCCESS * all_placed            # small terminal bonus
               + W_TABLE_PEN * touching_table      # safety
               - W_SMOOTH * ||a_t - a_{t-1}||      # damp oscillation for sim2real
        """
        if self.sparse_reward:
            # Pure per-cube placement: # of ACTIVE cubes currently in their slot (0..num_active).
            # No reach/grasp shaping, no negatives. Per-step (persists) so gamma=0.9 propagates.
            return (info["in_slot"] & self.active_mask).sum(dim=-1).float()
        batch = torch.arange(self.num_envs, device=self.device)
        cur_idx = info["current_color_idx"]

        # ---- Continuous staged shaping for the CURRENT cube ----
        reach = 1.0 - torch.tanh(REACH_TANH_K * info["tcp_to_current_dist"])     # 0..1
        is_grasp = info["is_grasping_current"].float()                          # 0/1
        cur_dist_to_slot = info["dist_cube_to_slot"][batch, cur_idx]            # (E,)
        # place is GATED by grasp: only rewarded for carrying the grasped cube toward its slot
        place = is_grasp * (1.0 - torch.tanh(PLACE_TANH_K * cur_dist_to_slot))  # 0..1
        staged = reach + W_GRASP_STAGE * is_grasp + place                      # 0..STAGED_MAX

        # ---- Monotone progress base (each placed cube worth KP > staged_max) ----
        # in_slot already requires z<PLACEMENT_Z_MAX (cube lowered to table at slot).
        in_slot = info["in_slot"] & self.active_mask
        self.ever_placed = self.ever_placed | in_slot
        num_placed = self.ever_placed.sum(dim=-1).float()                      # (E,)
        base = KP_PROGRESS * num_placed

        # ---- Small terminal + safety ----
        success_bonus = W_SUCCESS * info["success"].float()
        table_pen = W_TABLE_PEN * info["touching_table"].float()

        # ---- Action-smoothness penalty (sim2real: damp the real-arm oscillation) ----
        if not hasattr(self, "prev_action") or self.prev_action.shape != action.shape:
            self.prev_action = torch.zeros_like(action)
        smooth_pen = -W_SMOOTH * (action - self.prev_action).abs().mean(dim=-1)
        self.prev_action = action.detach().clone()

        return base + staged + success_bonus + table_pen + smooth_pen

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        if self.sparse_reward:
            # 0..1: fraction of active cubes currently placed (n=2 -> 0 / 0.5 / 1.0).
            n_placed = (info["in_slot"] & self.active_mask).sum(dim=-1).float()
            n_active = self.active_mask.sum(dim=-1).clamp(min=1).float()
            return n_placed / n_active
        # Scale to O(1)-per-cube. PPO normalizes advantages so absolute scale is not critical.
        return self.compute_dense_reward(obs=obs, action=action, info=info) / KP_PROGRESS


if __name__ == "__main__":
    # Smoke test
    import gymnasium as gym
    import mani_skill.envs  # noqa: register
    # Import our env so it gets registered
    print("Creating SortCubesSO100-v1...")
    env = gym.make(
        "SortCubesSO100-v1",
        num_envs=4,
        obs_mode="state",
        sim_backend="gpu",
        render_backend="cpu",
        num_active_cubes=5,
    )
    print(f"  action_space: {env.action_space}")
    print(f"  observation_space: {env.observation_space}")
    obs, info = env.reset(seed=0)
    print(f"  reset OK")
    print(f"  obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    print(f"  info keys: {list(info.keys())}")
    print(f"  success: {info['success']}")
    print(f"  current_color_idx: {info['current_color_idx']}")
    print(f"  n_placed: {info['n_placed']}")

    # Step 30 random actions
    import numpy as np
    total = torch.zeros(4)
    for i in range(30):
        a = torch.rand((4, env.action_space.shape[-1]), device="cuda") * 2 - 1
        obs, r, term, trunc, info = env.step(a)
        total += r.cpu()
        if i in (0, 15, 29):
            print(f"  step {i:2d}: r={r.cpu().tolist()}  "
                  f"n_placed={info['n_placed'].cpu().tolist()}")
    print(f"\nSmoke test OK, total reward per env = {total.tolist()}")
    env.close()
