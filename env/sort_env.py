"""Gymnasium env for the SO-ARM101 5-block strict-order sorting task in MuJoCo.

E2 (this file):
- Curriculum hook: `num_active_cubes` (1..5). Inactive cubes are parked off-camera at
  unreachable x so the arm physically cannot touch them.
- Strict order R→O→Y→G→B baked into the reward: only the lowest-color-index "not-placed"
  cube gives reward. Picking up the wrong cube does not pay.
- Reward components (shaped, all dense):
    R_reach     — when current cube not yet grasped, push tip toward it
    R_transport — when current cube is grasped, push it toward its slot
    R_grasp     — small per-step bonus while current cube is held
    R_placement — sparse +5 per cube transitioned into its slot
                  sparse -2 per cube knocked back out  (disorder penalty)
    R_done      — +10 when num_active cubes are all in their slots
    R_smooth    — -|Δa|²
    R_step      — -0.005
- Placement detection uses hysteresis (enter at 2.0 cm, exit at 2.5 cm) to avoid
  oscillation under contact jitter.
- State vector (25,): arm_qpos(6) + arm_qvel(6) + tip_xyz(3) + active_mask(5) +
  current_color_onehot(5).  Inactive cubes' coords still appear in `cube_positions` as
  their parked location, but the active_mask tells the policy which to look at.
- Images optional via `render_images` ctor flag — set False for state-only training to
  cut step latency by >10x.

Reward is computed from privileged sim state; nothing in `_compute_sort_reward` reads
images, so disabling cameras is safe for training.

NOT in this file:
- Domain randomization
- Vision encoders
- Algorithm (PPO/SAC) — separate file in next step
"""
from __future__ import annotations

import os
import pathlib
import re
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco


# ---------------- Task constants ----------------

CUBE_COLORS = ("red", "orange", "yellow", "green", "blue")
NUM_COLORS = len(CUBE_COLORS)

# Color index i → its target slot xy. Order is the fixed sort order.
TARGET_POSITIONS = np.array(
    [
        [0.30, -0.10],  # red
        [0.30, -0.05],  # orange
        [0.30,  0.00],  # yellow
        [0.30,  0.05],  # green
        [0.30,  0.10],  # blue
    ],
    dtype=np.float64,
)

# Cube spawn region for active cubes
SPAWN_X_RANGE = (0.12, 0.22)
SPAWN_Y_RANGE = (-0.12, 0.12)
CUBE_HALF_SIZE = 0.0125
CUBE_REST_Z = CUBE_HALF_SIZE + 0.001
MIN_PAIR_DIST = 0.045

# Parking position for inactive cubes (curriculum). Well outside arm reach (~30 cm)
# and outside front-camera FOV (camera at +0.55 m looking toward origin → +1.5 m is
# behind the camera).
PARK_X = 1.5
PARK_Y_OFFSETS = (-0.4, -0.2, 0.0, 0.2, 0.4)  # spread inactive cubes apart

# Home posture (rad). Gripper open, slight crouch above workspace.
HOME_QPOS_ARM = np.array(
    [0.0, -0.8, 1.2, -0.4, 0.0, 0.6],
    dtype=np.float64,
)

# Control: physics 200 Hz (timestep 0.005 from XML), policy ~28.6 Hz
N_SUBSTEPS = 7
DEFAULT_MAX_STEPS = 750  # ~26 s @ 28.6 Hz

# Placement detection (with hysteresis to avoid contact-jitter bouncing)
PLACEMENT_TOL_IN  = 0.020   # enter "placed" state at 2.0 cm
PLACEMENT_TOL_OUT = 0.025   # leave  "placed" state at 2.5 cm
PLACEMENT_Z_MAX   = 0.05    # must be near table (not held aloft)

# Grasp heuristic — current cube counts as grasped if tip is close AND cube is lifted.
GRASP_TIP_DIST_MAX = 0.04   # 4 cm tip-to-cube
GRASP_LIFT_Z_MIN   = 0.025  # cube center > 2.5 cm above floor → off the table

# Magnetic grasp params (kinematic attachment). When the gripper-close condition is
# met, the nearest in-range cube becomes "attached" to the tip site each step. This
# is the standard sim-RL approach used by Meta-World, Adroit, ManiSkill — exact
# contact physics on 25 mm cubes with the SO-ARM101 parallel jaws are unreliable
# in MuJoCo, so we abstract the grasp into a kinematic constraint. For sim2real,
# domain randomization on grasp tolerances will train the policy to handle real
# contact behavior.
MAGNETIC_GRASP_ENGAGE_GRIPPER_Q = 0.5   # gripper joint qpos below this → "trying to close"
MAGNETIC_GRASP_RELEASE_GRIPPER_Q = 0.8  # gripper joint qpos above this → "opening / release"
MAGNETIC_GRASP_DIST_MAX         = 0.04  # 4 cm tip-to-cube to engage
MAGNETIC_GRASP_LIFT_OFFSET      = 0.0   # cube center offset below tip when attached

# Reward weights — v4 ManiSkill-style dense (paired with gamma=0.9 in trainer).
#
# IMPORTANT iteration history:
#   v1 — tanh reach saturated → hover-near-cube farming
#   v2 — linear -dist reach + per-step W_GRASP_BONUS=1.0 + W_DONE=50
#        → hover-WITH-cube-near-slot farming (γ-discounted +1/step worth ~100 vs +50 done)
#   v3 — removed per-step grasp, added ONE-SHOT +20 grasp + +100 done. Worked for n=1
#        (100% success) but multi-cube plateaued: sparse one-shot signal too rare for SAC
#        to learn the orange-placement phase after red placement.
#   v4 (current) — ManiSkill-style: REPLACE one-shot with dense per-step bonus
#        is_grasping × (W_GRASP_KEEP + W_TRANSPORT_TANH × (1 - tanh(5×dist_to_slot))).
#        Bounded vs done at gamma=0.9: 3 / 0.1 = 30 < W_DONE=50. Provides continuous
#        gradient throughout transport phase — critical for compositional multi-cube.
# v4 attempt 2: REMOVE dense per-step bonuses (caused +573 hover farming).
# Keep v3 sparse one-shot scheme. The "ManiSkill recipe" doesn't transfer cleanly
# because their reward structure relies on episode termination + low gamma + their
# specific velocity-gated settle reward. For SO-ARM101 with current physics, sparse
# rewards are safer.
W_REACH         = 2.0   # × -dist(tip, current cube), only when NOT grasping (linear)
W_TRANSPORT     = 2.0   # × -dist(cube, slot), only when grasping (linear)
W_GRASP_KEEP    = 0.0   # DISABLED — dense version caused hover-farming (+573 in 60K)
W_TRANSPORT_TANH = 0.0  # DISABLED — same reason
W_GRASP_BONUS   = 0.0   # legacy
W_GRASP_ONESHOT = 20.0  # one-shot first grasp per color (v3 value)
W_ATTEMPT_GRASP = 0.0
W_PLACE_BONUS   = 10.0  # one-shot per cube placed (v3)
W_DISORDER      = -5.0  # one-shot per cube knocked out
W_DONE          = 100.0 # one-shot terminal (v3)
W_SMOOTH        = -0.02
W_STEP          = -0.01
ATTEMPT_GRASP_DIST_MAX     = 0.06  # unused in v4 but kept
ATTEMPT_GRASP_GRIPPER_MAX  = 0.5   # unused in v4 but kept

# Action mode: 'absolute' = action ∈ [-1,1]^6 mapped linearly to actuator ctrlrange (v1-v3);
#              'delta'    = action × MAX_DELTA added to current joint qpos each step (v4, ManiSkill style)
ACTION_MODE_DELTA_MAX = np.array(
    # per-joint max delta per policy step (rad). Smaller for finer joints, larger for big joints.
    [0.10, 0.08, 0.08, 0.10, 0.20, 0.40],  # gripper has wide range so larger delta is OK
    dtype=np.float64,
)


# ---------------- Env ----------------

class SortBlocksEnv(gym.Env):
    """SO-ARM101 strict-order sort: place R→O→Y→G→B into matching slots.

    Curriculum: pass `num_active_cubes` ∈ {1..5}. Only the first N colors are spawned
    in the workspace; the remaining cubes are parked outside reach.

    Action space: Box(-1, +1, (6,), float32). Mapped to actuator ctrlrange.

    Observation space (Dict):
        state          : (25,) arm_qpos(6) + arm_qvel(6) + tip_xyz(3)
                                + active_mask(5) + current_color_onehot(5)
        cube_positions : (5, 3) world xyz (inactive cubes show parked coords)
        image_front    : (H, W, 3) uint8  (only if render_images=True)
        image_wrist    : (H, W, 3) uint8  (only if render_images=True)

    Info dict (per step) keys:
        cube_xy, target_xy, cube_to_target_dist, placed_mask, sort_progress,
        active_mask, current_color_idx, is_grasping_current,
        tip_xyz, step, reward_components, is_success
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        xml_path: str | None = None,
        num_active_cubes: int = 5,
        max_steps: int = DEFAULT_MAX_STEPS,
        image_size: tuple[int, int] = (224, 224),
        render_images: bool = True,
        seed: int | None = None,
        fixed_cube_xys: list[tuple[float, float]] | None = None,
        randomize_active_color: bool = False,
        action_mode: str = "absolute",
    ):
        super().__init__()
        if not (1 <= num_active_cubes <= NUM_COLORS):
            raise ValueError(f"num_active_cubes must be in [1, {NUM_COLORS}]")
        self.num_active_cubes = int(num_active_cubes)
        # Optional fixed spawn — if set, cube xy positions are deterministic each reset.
        # Use for "easy mode" curriculum within stage 1 (learn grasp before generalizing).
        self.fixed_cube_xys = (
            np.array(fixed_cube_xys, dtype=np.float64) if fixed_cube_xys is not None else None
        )
        # Optional randomization: each episode picks a random color to be the active one
        # (only meaningful for num_active_cubes=1). Lets the warm-started policy at
        # stage 2+ have seen all 5 current_color_onehot patterns during stage 1 training.
        self.randomize_active_color = bool(randomize_active_color)
        # Action mode: 'absolute' (v1-v3) or 'delta' (v4, ManiSkill style — small joint deltas per step)
        if action_mode not in ("absolute", "delta"):
            raise ValueError(f"action_mode must be 'absolute' or 'delta', got {action_mode!r}")
        self.action_mode = action_mode

        if xml_path is None:
            here = pathlib.Path(__file__).resolve().parent
            xml_path = str((here.parent / "sim" / "scene_sort.xml").resolve())
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"scene XML not found: {xml_path}")

        self._xml_path = xml_path
        # MuJoCo resolves <include>'s meshdir relative to the parent file, not the
        # included file. We inline the include + rewrite meshdir to an absolute path,
        # so we never touch the menagerie source on disk.
        xml_str = _preprocess_scene_xml(xml_path)
        # menagerie's sts3215 actuator class declares forcerange="-2.94 2.94" Nm, which
        # is not enough to hold the arm against gravity at full extension (low-z reach).
        # We patch the inline copy here (NOT the source file) to use a generous forcerange.
        # Mass-balanced: real STS3215 servo can deliver ~30 kg-cm ≈ 2.94 Nm static torque
        # but our sim's joint dynamics include arm inertia + gravity → we need headroom.
        xml_str = xml_str.replace('forcerange="-2.94 2.94"', 'forcerange="-10 10"')
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        self.max_steps = int(max_steps)
        self.image_h, self.image_w = int(image_size[0]), int(image_size[1])
        self.render_images = bool(render_images)

        # Pre-resolve names → ids
        self._arm_joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
        self._arm_joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
             for n in self._arm_joint_names], dtype=np.int32
        )
        self._arm_qpos_adr = np.array(
            [self.model.jnt_qposadr[i] for i in self._arm_joint_ids], dtype=np.int32
        )
        self._arm_qvel_adr = np.array(
            [self.model.jnt_dofadr[i] for i in self._arm_joint_ids], dtype=np.int32
        )

        self._cube_body_ids = []
        self._cube_jnt_qpos_adr = []
        for c in CUBE_COLORS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{c}_free")
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{c}")
            if jid < 0 or bid < 0:
                raise RuntimeError(f"missing cube joint/body for color={c}")
            self._cube_jnt_qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            self._cube_body_ids.append(int(bid))
        self._cube_jnt_qpos_adr = np.array(self._cube_jnt_qpos_adr, dtype=np.int32)
        self._cube_body_ids = np.array(self._cube_body_ids, dtype=np.int32)

        self._tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        if self._tip_site_id < 0:
            raise RuntimeError("expected site 'gripperframe' on the gripper body")

        self._cam_front_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam"
        )
        self._cam_wrist_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
        )

        self._ctrl_lo = self.model.actuator_ctrlrange[:6, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:6, 1].copy()

        self._renderer: mujoco.Renderer | None = None

        # Spaces — state is now 25-dim
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        state_dim = 6 + 6 + 3 + NUM_COLORS + NUM_COLORS + 3 + 2  # = 30 (with current cube + slot explicit)
        obs_spaces: dict[str, spaces.Space] = {
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
            "cube_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_COLORS, 3), dtype=np.float32),
        }
        if self.render_images:
            obs_spaces["image_front"] = spaces.Box(low=0, high=255,
                                                  shape=(self.image_h, self.image_w, 3), dtype=np.uint8)
            obs_spaces["image_wrist"] = spaces.Box(low=0, high=255,
                                                  shape=(self.image_h, self.image_w, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

        # Gripper joint qpos index (last of the arm joints in our arm_joint_names list)
        self._gripper_qpos_adr = int(self._arm_qpos_adr[5])

        # Per-episode state
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._last_action = np.zeros(6, dtype=np.float32)
        self._placed_mask = np.zeros(NUM_COLORS, dtype=bool)
        self._active_mask = np.zeros(NUM_COLORS, dtype=bool)
        # Magnetic grasp state: index of currently-attached cube (-1 = none)
        self._held_cube_idx = -1
        # One-shot tracking: set of color indices for which the grasp_oneshot has fired
        self._grasp_oneshot_fired = np.zeros(NUM_COLORS, dtype=bool)
        # Once placed flag — placement bonus only fires the FIRST time a cube enters its
        # slot in an episode. Without this, the policy can farm +5 per in/out cycle
        # (+10 place, -5 disorder = +5 net). Disorder penalty still fires per removal.
        self._ever_placed_mask = np.zeros(NUM_COLORS, dtype=bool)

    # ---------- core API ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Curriculum override via options dict at runtime (handy for callbacks)
        if options and "num_active_cubes" in options:
            n = int(options["num_active_cubes"])
            if 1 <= n <= NUM_COLORS:
                self.num_active_cubes = n

        mujoco.mj_resetData(self.model, self.data)

        # Home posture
        self.data.qpos[self._arm_qpos_adr] = HOME_QPOS_ARM
        self.data.qvel[self._arm_qvel_adr] = 0.0
        self.data.ctrl[:6] = HOME_QPOS_ARM.astype(np.float32)

        # Active mask: first num_active_cubes colors (or randomized colors if requested)
        self._active_mask = np.zeros(NUM_COLORS, dtype=bool)
        if self.randomize_active_color and self.num_active_cubes < NUM_COLORS:
            # Randomly pick num_active distinct colors for this episode
            chosen = self._rng.choice(NUM_COLORS, size=self.num_active_cubes, replace=False)
            self._active_mask[chosen] = True
        else:
            self._active_mask[:self.num_active_cubes] = True
        self._placed_mask = np.zeros(NUM_COLORS, dtype=bool)

        # Spawn active cubes in workspace; park inactive ones out of reach.
        # When randomize_active_color is on, we need to assign positions to the
        # actually-active color indices (which may not be 0..n-1).
        active_indices = np.where(self._active_mask)[0]
        active_xys = np.zeros((NUM_COLORS, 2), dtype=np.float64)
        if self.fixed_cube_xys is not None:
            for i, ai in enumerate(active_indices):
                if i < len(self.fixed_cube_xys):
                    active_xys[ai] = self.fixed_cube_xys[i]
        else:
            sampled = self._sample_cube_xys(self.num_active_cubes)
            for i, ai in enumerate(active_indices):
                active_xys[ai] = sampled[i]
        for i, base in enumerate(self._cube_jnt_qpos_adr):
            if self._active_mask[i]:
                x, y, z = float(active_xys[i, 0]), float(active_xys[i, 1]), CUBE_REST_Z
                yaw = float(self._rng.uniform(-np.pi, np.pi))
                qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
                quat = (qw, 0.0, 0.0, qz)
            else:
                x, y, z = PARK_X, PARK_Y_OFFSETS[i], CUBE_REST_Z
                quat = (1.0, 0.0, 0.0, 0.0)
            self.data.qpos[base + 0] = x
            self.data.qpos[base + 1] = y
            self.data.qpos[base + 2] = z
            self.data.qpos[base + 3] = quat[0]
            self.data.qpos[base + 4] = quat[1]
            self.data.qpos[base + 5] = quat[2]
            self.data.qpos[base + 6] = quat[3]

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._last_action[:] = 0.0
        self._held_cube_idx = -1
        self._grasp_oneshot_fired[:] = False
        self._ever_placed_mask[:] = False

        obs = self._get_obs()
        info = self._get_info(reward_components={})
        return obs, info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.action_mode == "delta":
            # Delta-joint mode (ManiSkill style): action × MAX_DELTA added to current joint qpos.
            # Smaller-step incremental control is easier for SAC/PPO to learn than absolute targets.
            current_q = self.data.qpos[self._arm_qpos_adr].astype(np.float64)
            target_q = current_q + a.astype(np.float64) * ACTION_MODE_DELTA_MAX
            ctrl = np.clip(target_q, self._ctrl_lo, self._ctrl_hi)
        else:
            # Absolute mode: action ∈ [-1,1] mapped linearly to actuator ctrlrange
            ctrl = self._ctrl_lo + (a + 1.0) * 0.5 * (self._ctrl_hi - self._ctrl_lo)
        self.data.ctrl[:6] = ctrl
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # Magnetic grasp resolution — after physics, before reward
        self._resolve_magnetic_grasp()

        # Update placement mask (with hysteresis) BEFORE building obs/reward so they agree
        cube_pos = self._cube_world_pos()
        new_placed = self._update_placed_mask(cube_pos)

        reward, components = self._compute_sort_reward(
            cube_pos=cube_pos, action=a, last_action=self._last_action,
            prev_placed=self._placed_mask, new_placed=new_placed,
        )
        self._placed_mask = new_placed

        obs = self._get_obs()
        info = self._get_info(reward_components=components)
        is_success = bool((self._placed_mask & self._active_mask).sum() == self.num_active_cubes)
        info["is_success"] = is_success

        terminated = is_success
        truncated = bool(self._step_count >= self.max_steps)

        self._last_action = a.copy()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self._render_cam(self._cam_front_id)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---------- helpers ----------

    def _sample_cube_xys(self, n: int) -> np.ndarray:
        out = np.zeros((NUM_COLORS, 2), dtype=np.float64)  # full 5 slots, but only first n are used
        for i in range(n):
            for _attempt in range(200):
                x = float(self._rng.uniform(*SPAWN_X_RANGE))
                y = float(self._rng.uniform(*SPAWN_Y_RANGE))
                if i == 0:
                    out[i] = (x, y); break
                d = np.linalg.norm(out[:i] - np.array([x, y]), axis=1).min()
                if d >= MIN_PAIR_DIST:
                    out[i] = (x, y); break
            else:
                out[i] = out[i - 1] + self._rng.normal(0, 0.05, size=2)
        return out

    def _cube_world_pos(self) -> np.ndarray:
        return np.array(
            [self.data.xpos[bid] for bid in self._cube_body_ids],
            dtype=np.float64,
        )

    def _resolve_magnetic_grasp(self) -> None:
        """Engage / release the kinematic cube-tip attachment based on gripper qpos.

        Engagement rules:
          - If currently holding none AND gripper qpos < ENGAGE threshold AND a cube
            (active only) is within DIST_MAX of tip → attach the closest such cube.
          - If currently holding cube i AND gripper qpos > RELEASE threshold → detach.

        While attached, the cube's freejoint qpos is overwritten each step to track
        the tip site position (with zero rotation for simplicity). Cube qvel is zeroed
        so it doesn't fly off when released. This is a kinematic override applied AFTER
        mj_step, so the physics engine sees the new state on the next step.
        """
        gripper_q = float(self.data.qpos[self._gripper_qpos_adr])
        tip_xyz = self.data.site_xpos[self._tip_site_id].copy()

        # Release check — when releasing, snap cube down to floor at its current xy,
        # zero velocity. Prevents cube from inheriting arm's motion at the moment of
        # release and sliding off the slot.
        if self._held_cube_idx >= 0 and gripper_q > MAGNETIC_GRASP_RELEASE_GRIPPER_Q:
            i = self._held_cube_idx
            base = int(self._cube_jnt_qpos_adr[i])
            # Keep current xy but drop z to resting
            self.data.qpos[base + 2] = CUBE_REST_Z
            qveladr = int(self.model.jnt_dofadr[
                int(self.model.body_jntadr[self._cube_body_ids[i]])
            ])
            self.data.qvel[qveladr:qveladr + 6] = 0.0
            self._held_cube_idx = -1
            mujoco.mj_forward(self.model, self.data)

        # Engage check
        if self._held_cube_idx < 0 and gripper_q < MAGNETIC_GRASP_ENGAGE_GRIPPER_Q:
            best_i = -1
            best_d = MAGNETIC_GRASP_DIST_MAX
            for i in range(NUM_COLORS):
                if not self._active_mask[i]:
                    continue
                cube_xyz = self.data.xpos[self._cube_body_ids[i]]
                d = float(np.linalg.norm(cube_xyz - tip_xyz))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                self._held_cube_idx = best_i

        # If holding, slave cube qpos to tip
        if self._held_cube_idx >= 0:
            i = self._held_cube_idx
            base = int(self._cube_jnt_qpos_adr[i])
            self.data.qpos[base + 0] = tip_xyz[0]
            self.data.qpos[base + 1] = tip_xyz[1]
            # Keep cube center at tip; bias slightly upward so cube doesn't intersect floor
            self.data.qpos[base + 2] = max(float(tip_xyz[2]) - MAGNETIC_GRASP_LIFT_OFFSET, CUBE_REST_Z)
            self.data.qpos[base + 3] = 1.0
            self.data.qpos[base + 4] = 0.0
            self.data.qpos[base + 5] = 0.0
            self.data.qpos[base + 6] = 0.0
            # Zero cube velocity (we override qpos directly each step)
            qveladr = int(self.model.jnt_dofadr[
                int(self.model.body_jntadr[self._cube_body_ids[i]])
            ])
            self.data.qvel[qveladr:qveladr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def _update_placed_mask(self, cube_pos: np.ndarray) -> np.ndarray:
        """Apply hysteresis: enter placed at d<TOL_IN, leave at d>TOL_OUT.

        Only active cubes can be placed. Disorder (knock-out) happens when previously-placed
        cube drifts past TOL_OUT.
        """
        d = np.linalg.norm(cube_pos[:, :2] - TARGET_POSITIONS, axis=1)
        z_ok = cube_pos[:, 2] < PLACEMENT_Z_MAX
        new_mask = self._placed_mask.copy()
        for i in range(NUM_COLORS):
            if not self._active_mask[i]:
                new_mask[i] = False
                continue
            if self._placed_mask[i]:
                # was placed — stays placed unless d exceeds TOL_OUT or z above table
                new_mask[i] = bool(d[i] < PLACEMENT_TOL_OUT and z_ok[i])
            else:
                # was not placed — becomes placed only when crossing TOL_IN
                new_mask[i] = bool(d[i] < PLACEMENT_TOL_IN and z_ok[i])
        return new_mask

    def _current_color_idx(self) -> int:
        """Lowest active color index that isn't placed yet. -1 if all done."""
        unfinished = self._active_mask & ~self._placed_mask
        if not unfinished.any():
            return -1
        return int(np.argmax(unfinished))

    def _get_obs(self) -> dict[str, Any]:
        cube_pos = self._cube_world_pos().astype(np.float32)
        arm_qpos = self.data.qpos[self._arm_qpos_adr].astype(np.float32)
        arm_qvel = self.data.qvel[self._arm_qvel_adr].astype(np.float32)
        tip_xyz = self.data.site_xpos[self._tip_site_id].astype(np.float32)

        current_idx = self._current_color_idx()
        current_onehot = np.zeros(NUM_COLORS, dtype=np.float32)
        # Explicit "current cube position" — saves the policy from learning to
        # attend cube_positions via current_color_onehot. Set to zeros when done.
        current_cube_xyz = np.zeros(3, dtype=np.float32)
        current_slot_xy = np.zeros(2, dtype=np.float32)
        if current_idx >= 0:
            current_onehot[current_idx] = 1.0
            current_cube_xyz = cube_pos[current_idx].astype(np.float32)
            current_slot_xy = TARGET_POSITIONS[current_idx].astype(np.float32)

        state = np.concatenate([
            arm_qpos,         # 6
            arm_qvel,         # 6
            tip_xyz,          # 3
            self._active_mask.astype(np.float32),  # 5
            current_onehot,                         # 5
            current_cube_xyz, # 3 — NEW: explicit "what cube to target"
            current_slot_xy,  # 2 — NEW: explicit "where to place it"
        ])  # total = 30
        obs = {"state": state, "cube_positions": cube_pos}
        if self.render_images:
            obs["image_front"] = self._render_cam(self._cam_front_id)
            obs["image_wrist"] = self._render_cam(self._cam_wrist_id)
        return obs

    def _get_info(self, *, reward_components: dict) -> dict:
        cube_pos = self._cube_world_pos()
        d = np.linalg.norm(cube_pos[:, :2] - TARGET_POSITIONS, axis=1)
        cur_idx = self._current_color_idx()
        is_grasping_current = False
        if cur_idx >= 0:
            tip = self.data.site_xpos[self._tip_site_id]
            dist_tip = float(np.linalg.norm(tip - cube_pos[cur_idx]))
            is_grasping_current = (dist_tip < GRASP_TIP_DIST_MAX) and (cube_pos[cur_idx, 2] > GRASP_LIFT_Z_MIN)
        return {
            "cube_xy": cube_pos[:, :2].astype(np.float32),
            "target_xy": TARGET_POSITIONS.astype(np.float32),
            "cube_to_target_dist": d.astype(np.float32),
            "placed_mask": self._placed_mask.copy(),
            "active_mask": self._active_mask.copy(),
            "current_color_idx": cur_idx,
            "is_grasping_current": bool(is_grasping_current or self._held_cube_idx == cur_idx),
            "held_cube_idx": int(self._held_cube_idx),
            "sort_progress": int((self._placed_mask & self._active_mask).sum()),
            "tip_xyz": self.data.site_xpos[self._tip_site_id].astype(np.float32),
            "step": self._step_count,
            "reward_components": reward_components,
        }

    def _compute_sort_reward(
        self,
        *,
        cube_pos: np.ndarray,
        action: np.ndarray,
        last_action: np.ndarray,
        prev_placed: np.ndarray,
        new_placed: np.ndarray,
        tip_xyz: np.ndarray | None = None,
    ) -> tuple[float, dict]:
        """Strict-order shaped reward — see module docstring for breakdown.

        `tip_xyz` is an optional override (default: read from sim state). Test scripts
        pass custom tips to probe reward curves along scripted trajectories.
        """
        tip = self.data.site_xpos[self._tip_site_id] if tip_xyz is None else np.asarray(tip_xyz)

        # Determine the "current" cube — lowest active not-placed in the NEW state
        active = self._active_mask
        unfinished = active & ~new_placed
        if unfinished.any():
            cur_idx = int(np.argmax(unfinished))
            current_cube = cube_pos[cur_idx]
            current_target = TARGET_POSITIONS[cur_idx]
            dist_tip_cube = float(np.linalg.norm(tip - current_cube))
            dist_cube_target = float(np.linalg.norm(current_cube[:2] - current_target))
            # "Is currently grasping" — magnetic attachment of THIS cube,
            # OR fallback height heuristic (in case magnetic disabled / cube held differently)
            is_grasping = (self._held_cube_idx == cur_idx) or (
                (dist_tip_cube < GRASP_TIP_DIST_MAX) and (current_cube[2] > GRASP_LIFT_Z_MIN)
            )
        else:
            cur_idx = -1
            current_cube = None
            current_target = None
            dist_tip_cube = 0.0
            dist_cube_target = 0.0
            is_grasping = False

        # ----- Components (v4 ManiSkill-style dense) -----
        # Reach: when NOT grasping current cube, push tip toward it (linear -dist)
        if cur_idx >= 0 and not is_grasping:
            r_reach = -W_REACH * dist_tip_cube
        else:
            r_reach = 0.0

        # Transport: when grasping current cube, push cube toward slot.
        # Combines linear -dist (always active for direction) with tanh-shaped per-step
        # bonus (peaks at slot, provides continuous reward signal — ManiSkill PickCube style).
        if cur_idx >= 0 and is_grasping:
            r_transport = -W_TRANSPORT * dist_cube_target  # linear pull toward slot
            # Dense grasping bonus: W_GRASP_KEEP (constant while holding) + tanh-shaped
            # transport peak (max W_TRANSPORT_TANH when at slot).
            r_grasp_keep = W_GRASP_KEEP + W_TRANSPORT_TANH * (1.0 - float(np.tanh(5.0 * dist_cube_target)))
        else:
            r_transport = 0.0
            r_grasp_keep = 0.0

        # Backward-compat fields
        r_grasp_bonus = 0.0       # not used in v4
        r_attempt_grasp = 0.0     # disabled

        # Small one-shot grasp bonus (re-enabled at +5 to give discrete grasp signal)
        r_grasp_oneshot = 0.0
        if cur_idx >= 0 and is_grasping and not self._grasp_oneshot_fired[cur_idx]:
            r_grasp_oneshot = W_GRASP_ONESHOT
            self._grasp_oneshot_fired[cur_idx] = True

        # Placement delta — placement bonus only counts cubes that have NEVER been placed
        # before in this episode (prevents in/out farming). Disorder penalty fires per
        # removal regardless (to discourage knocking-out work in progress).
        # `added_new` ignores re-placement after a knock-out.
        added_new = int((new_placed & ~prev_placed & ~self._ever_placed_mask & active).sum())
        removed = int((~new_placed & prev_placed & active).sum())
        r_placement = W_PLACE_BONUS * float(added_new) + W_DISORDER * float(removed)
        # Update ever_placed_mask AFTER computing reward
        self._ever_placed_mask |= new_placed

        # Done bonus is ONE-SHOT: fires only on the step where the last cube transitions
        # into placed state. env.step() also terminates on success so this rarely matters
        # in real training, but it keeps cumulative-reward diagnostics interpretable.
        all_done_now = ((new_placed & active).sum() == int(active.sum()))
        all_done_prev = ((prev_placed & active).sum() == int(active.sum()))
        r_done = W_DONE if (all_done_now and not all_done_prev) else 0.0

        r_smooth = W_SMOOTH * float(np.sum((action - last_action) ** 2))
        r_step = W_STEP

        total = (r_reach + r_transport + r_grasp_keep + r_grasp_bonus + r_grasp_oneshot
                 + r_attempt_grasp + r_placement + r_done + r_smooth + r_step)

        comps = {
            "reach":        float(r_reach),
            "transport":    float(r_transport),
            "grasp_keep":   float(r_grasp_keep),
            "grasp_bonus":  float(r_grasp_bonus),
            "grasp_oneshot": float(r_grasp_oneshot),
            "attempt_grasp": float(r_attempt_grasp),
            "placement":    float(r_placement),
            "placement_added":   int(added_new),
            "placement_removed": int(removed),
            "done":         float(r_done),
            "smoothness":   float(r_smooth),
            "step_penalty": float(r_step),
            "total":        float(total),
            # Diagnostics
            "current_color_idx":   int(cur_idx),
            "dist_tip_to_current": float(dist_tip_cube),
            "dist_current_to_slot": float(dist_cube_target),
            "is_grasping_current": int(is_grasping),
            "n_placed": int((new_placed & active).sum()),
        }
        return total, comps

    def _render_cam(self, cam_id: int) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.image_h, width=self.image_w)
        self._renderer.update_scene(self.data, camera=cam_id)
        return self._renderer.render().copy()


# ---------------- XML preprocessing ----------------

_INCLUDE_RE = re.compile(r'<include\s+file\s*=\s*"([^"]+)"\s*/>')
_MESHDIR_RE = re.compile(r'meshdir\s*=\s*"[^"]*"')
_BODY_RE = re.compile(r"<mujoco[^>]*>(.*)</mujoco>\s*\Z", re.DOTALL)


def _preprocess_scene_xml(scene_path: str) -> str:
    """Inline every <include file="..."/> and rewrite meshdir to an absolute path.

    MuJoCo resolves the included file's meshdir relative to the *parent* XML, not the
    included file. That breaks when our scene lives in a different folder from the
    menagerie model. We do a textual inline + meshdir rewrite so meshes load correctly,
    and we never touch the menagerie source on disk.
    """
    scene_dir = pathlib.Path(scene_path).parent.resolve()
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = f.read()

    def _replace(match: re.Match) -> str:
        rel_or_abs = match.group(1)
        inc_path = pathlib.Path(rel_or_abs)
        if not inc_path.is_absolute():
            inc_path = (scene_dir / inc_path).resolve()
        if not inc_path.is_file():
            raise FileNotFoundError(f"include target missing: {inc_path}")
        with open(inc_path, "r", encoding="utf-8") as fi:
            inc_text = fi.read()
        abs_mesh_dir = (inc_path.parent / "assets").resolve().as_posix() + "/"
        inc_text = _MESHDIR_RE.sub(f'meshdir="{abs_mesh_dir}"', inc_text)
        body = _BODY_RE.search(inc_text)
        if not body:
            raise RuntimeError(f"could not extract <mujoco> body from {inc_path}")
        return body.group(1)

    return _INCLUDE_RE.sub(_replace, scene)


# ---------------- Convenience factory ----------------

def make_env(**kwargs) -> SortBlocksEnv:
    return SortBlocksEnv(**kwargs)
