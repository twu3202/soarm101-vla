"""SO-ARM101 5-cube sorting — real-arm deployment.

Loads a trained state-policy checkpoint and runs it on the physical follower arm.
No leader arm needed. References D:\\soarm101\\rescue_record.py for lerobot 0.2+ API.

Pipeline per step:
    [USB camera frame]   → HSV color seg → 5 cube xyz (in robot frame)
    [Follower servos]    → present qpos (deg) → radians + FK → tcp_pos
    → build 42-dim state → policy(state) → 6-d action in [-1,1]
    → action × controller_scale → delta_qpos → target_qpos accumulator
    → target_qpos in radians → degrees → send_action dict

Action interpretation (matches SO100GraspCube sim controller "pd_joint_target_delta_pos"):
    action_range = [-0.05, +0.05] rad for joints 0..4 (arm), [-0.2, +0.2] rad for joint 5 (gripper)
    target_qpos[t+1] = target_qpos[t] + action × range
    NOTE: target accumulates from previous TARGET (not measured qpos).

Usage:
    # 1. Calibrate camera intrinsic (one-time, with 9x6 25mm chessboard)
    python deploy_real.py --calibrate_camera --camera_index=0

    # 2. Calibrate camera→robot extrinsic via solvePnP (lay board flat in
    #    workspace, measure 3 highlighted corners' robot-frame X,Y with a ruler).
    #    Recovers full T_robot_cam — robust to camera mounting tilt, no angles.
    python deploy_real.py --calibrate_extrinsic --camera_index=0 --square_size=0.025
    #    (legacy manual fallback: python build_T_robot_cam.py — error-prone, not recommended)

    # 3. Tune HSV thresholds
    python deploy_real.py --tune_hsv --camera_index=0

    # 4. Deploy with policy
    python deploy_real.py --ckpt=runs/sort_n5_v16_DRheavy/ckpt_101.pt \\
        --robot_port=COM4 --camera_index=0 --calibration=calib.npz
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# === Constants matching sort_cubes_env.py ===
# Slot row adopted from the user's OWN teleop-demo layout (measured realized means, std ~1.5cm):
# red on the LEFT (+y), blue on the RIGHT (-y), at x~0.32. This is the MIRROR (in y) of the old
# 0.28-row the RL policy was trained on — so the env+policy are being retrained to this layout
# and the demos (which place here) become self-consistent. If you ever revert, the old row was
# x=0.28, y=[-0.10,-0.05,0.0,0.05,0.10].
SLOT_XY = np.array([
    [0.32,  0.09],  # red    (user demo: front-LEFT)
    [0.32,  0.03],  # orange
    [0.32, -0.03],  # yellow
    [0.32, -0.09],  # green
    [0.32, -0.14],  # blue   (front-RIGHT)
], dtype=np.float32)
PLACEMENT_XY_TOL = 0.035  # loosened from 0.025: demo placements have ~1.5cm std; slots 6cm apart
PLACEMENT_Z_MAX = 0.05
CUBE_TABLE_Z = 0.013

# Detection sanity gates (fix: wide FOV picked up the operator's arm/skin at frame edges as a
# phantom cube, teleporting a detection ~60cm). A blob is accepted only if BOTH hold:
#   (1) its centroid lies inside the central DETECT_ROI_FRAC of the image (drop edge clutter), and
#   (2) its back-projected world xy lies inside DETECT_WORKSPACE_BOX (reachable table region).
DETECT_ROI_FRAC = 0.80                          # keep central 80% of width/height; tune via --roi_frac
DETECT_WORKSPACE_BOX = (0.08, 0.42, -0.28, 0.28)  # (xmin, xmax, ymin, ymax) in robot/world frame

# Sim WORLD-frame origin = URDF Base link, which sits 4.52cm BEHIND the shoulder_pan
# (J0) vertical axis along +X (verified via FK circle-fit + URDF joint origin).
# The policy/tcp/cube/slot coords all live in this world frame. If you measure
# camera-calibration points from the (visible) shoulder_pan axis, add this offset
# to convert them into the world frame. Y offset is 0.
SHOULDER_PAN_WORLD_XY = np.array([0.0452, 0.0], dtype=np.float64)
N_JOINTS = 6
WRIST_ROLL_JOINT = 4  # fixed to 0 by sim policy
CONTROL_HZ = 15  # lerobot-sim2real recommends ≤15Hz for safety

# Action-to-delta mapping from SO100GraspCube's pd_joint_target_delta_pos controller
# action[i] ∈ [-1, 1] → delta_qpos[i] = action[i] * ACTION_RANGE[i]
ACTION_RANGE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.2], dtype=np.float32)

# Rest pose (matches sim training rest_qpos)
# wrist_roll (index 4) locked at -19.3 deg (user's "left-side / leader-default" pose) instead of
# 90 deg (camera-down). Override at runtime with --wrist_roll_deg.
REST_QPOS = np.array([0, 0, 0, np.pi / 2, np.deg2rad(-19.3), 0], dtype=np.float32)

# SO-100/101 joint names (lerobot convention)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# Elbow_flex calibration offset (degrees) — see ManiSkill LeRobotRealAgent
ELBOW_FLEX_OFFSET_DEG = 6.8


# =====================================================================
# Policy (matches ppo.py Agent)
# =====================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class StateAgent(nn.Module):
    def __init__(self, state_dim=42, act_dim=6):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    @torch.no_grad()
    def get_action(self, state):
        return self.actor_mean(state)


# =====================================================================
# Forward kinematics — pytorch_kinematics with SO-100 URDF
# =====================================================================
class SO100ForwardKinematics:
    """FK using pytorch_kinematics. Returns TCP = midpoint of Fixed_Jaw_tip / Moving_Jaw_tip."""

    def __init__(self, urdf_path: str = None):
        try:
            import pytorch_kinematics as pk
        except ImportError:
            raise RuntimeError("Install: pip install pytorch_kinematics")
        if urdf_path is None:
            import mani_skill
            urdf_path = os.path.join(
                os.path.dirname(mani_skill.__file__),
                "assets", "robots", "so100", "so100.urdf",
            )
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        with open(urdf_path, "rb") as f:
            self.chain_fix = pk.build_serial_chain_from_urdf(f.read(), "Fixed_Jaw_tip")
        with open(urdf_path, "rb") as f:
            self.chain_mov = pk.build_serial_chain_from_urdf(f.read(), "Moving_Jaw_tip")
        # Active joints in chain (verify):
        print(f"FK loaded URDF: {urdf_path}")
        print(f"  Fixed_Jaw_tip joints: {self.chain_fix.get_joint_parameter_names()}")
        print(f"  Moving_Jaw_tip joints: {self.chain_mov.get_joint_parameter_names()}")

    def tcp_pos(self, qpos_rad: np.ndarray) -> np.ndarray:
        """qpos_rad: (6,) — full 6-DOF joint angles. Returns (3,) TCP xyz in WORLD frame.
        Fixed_Jaw_tip chain uses 5 joints (no gripper), Moving_Jaw_tip uses 6.
        Sim env rotates robot base 90° around Z (yaw=π/2), so we apply same rotation
        to URDF-frame FK output to match sim's world-frame state convention.
        """
        q5 = torch.tensor(qpos_rad[:5], dtype=torch.float32)
        q6 = torch.tensor(qpos_rad, dtype=torch.float32)
        tm_fix = self.chain_fix.forward_kinematics(q5)
        tm_mov = self.chain_mov.forward_kinematics(q6)
        p_fix = tm_fix.get_matrix()[0, :3, 3]
        p_mov = tm_mov.get_matrix()[0, :3, 3]
        tcp_urdf = ((p_fix + p_mov) / 2).cpu().numpy().astype(np.float32)
        # Rotate from URDF base frame to world frame (sim's Pose has q=euler2quat(0,0,π/2)):
        # R_z(π/2) maps URDF (x,y,z) → world (-y, x, z)
        tcp_world = np.array([-tcp_urdf[1], tcp_urdf[0], tcp_urdf[2]], dtype=np.float32)
        return tcp_world


# =====================================================================
# Cube tracker — HSV color segmentation
# =====================================================================
class CubeTracker:
    # Tuned to measured cube hues: red H~178, orange H~4, yellow H~19, green H~55, blue H~104.
    # Orange is very red (H~4) so it takes the low warm band; red keeps only the high-wrap band
    # to avoid stealing the orange cube. Boundaries: orange|yellow~11, yellow|green~30.
    COLOR_HSV_RANGES = {
        # Red wraps the 0/180 hue boundary — it reads ~178-180 AND ~0-4. Cover BOTH bands,
        # else it flickers/drops when the hue crosses 0 (measured: H jumps 178<->1, S/V high).
        "red":    [((0, 80, 50), (2, 255, 255)), ((160, 80, 50), (180, 255, 255))],
        "orange": [((3, 90, 60),  (11, 255, 255))],
        "yellow": [((12, 80, 60),  (30, 255, 255))],
        "green":  [((31, 60, 50),  (82, 255, 255))],
        "blue":   [((85, 90, 60),  (135, 255, 255))],
    }
    COLOR_ORDER = ["red", "orange", "yellow", "green", "blue"]

    def __init__(self, camera_index=0, K=None, dist=None, T_robot_cam=None,
                 z_table=CUBE_TABLE_Z, min_blob_area=50, pos_offset=None,
                 roi_frac=DETECT_ROI_FRAC, workspace_box=DETECT_WORKSPACE_BOX):
        import cv2
        self.cv2 = cv2
        # Use DSHOW backend on Windows (MSMF has known phantom-device / read-fail issues)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        self.K = K
        self.dist = dist  # lens distortion coeffs; if set, pixels are undistorted before ray-casting
        self.T_robot_cam = T_robot_cam
        self.roi_frac = float(roi_frac)          # central image fraction kept (1.0 = whole frame)
        self.workspace_box = workspace_box       # (xmin,xmax,ymin,ymax) world-frame accept region
        # Constant (dx, dy) world-frame correction added to every cube xy. Corrects a
        # pure-translation calibration bias (mis-located anchor corner) without re-shooting.
        self.pos_offset = (np.zeros(2, dtype=np.float32) if pos_offset is None
                           else np.asarray(pos_offset, dtype=np.float32).reshape(2))
        self.z_table = float(z_table)
        self.min_blob_area = int(min_blob_area)
        # Sensible prior positions (spread along workspace y)
        self.last_positions = np.zeros((5, 3), dtype=np.float32)
        self.last_positions[:, 0] = 0.27
        self.last_positions[:, 1] = np.linspace(-0.12, 0.12, 5)
        self.last_positions[:, 2] = CUBE_TABLE_Z
        self.last_detected = np.zeros(5, dtype=bool)  # which cubes were FRESHLY seen last frame

    def _pixel_to_world(self, u, v):
        if self.K is None or self.T_robot_cam is None:
            raise RuntimeError("Camera not calibrated")
        if self.dist is not None:
            # Undistort: returns normalized cam-frame coords (K_inv already applied).
            und = self.cv2.undistortPoints(
                np.array([[[float(u), float(v)]]], dtype=np.float32), self.K, self.dist
            ).reshape(2)
            ray_cam = np.array([und[0], und[1], 1.0])
        else:
            ray_cam = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        R = self.T_robot_cam[:3, :3]
        t = self.T_robot_cam[:3, 3]
        ray_world = R @ ray_cam
        if abs(ray_world[2]) < 1e-6:
            return self.last_positions[0].copy()
        s = (self.z_table - t[2]) / ray_world[2]
        return (t + s * ray_world).astype(np.float32)

    def get_cube_positions(self, return_debug=False):
        ret, frame = self.cap.read()
        if not ret:
            return (self.last_positions.copy(), None, []) if return_debug else self.last_positions.copy()
        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]
        # Central ROI rectangle: ignore edge clutter (operator's arm/skin at frame border).
        rx0, rx1 = int(W * (1 - self.roi_frac) / 2), int(W * (1 + self.roi_frac) / 2)
        ry0, ry1 = int(H * (1 - self.roi_frac) / 2), int(H * (1 + self.roi_frac) / 2)
        bx0, bx1, by0, by1 = self.workspace_box
        out = np.zeros((5, 3), dtype=np.float32)
        detected = np.zeros(5, dtype=bool)
        debug = []
        for ci, color in enumerate(self.COLOR_ORDER):
            mask = None
            for lo, hi in self.COLOR_HSV_RANGES[color]:
                m = self.cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = m if mask is None else self.cv2.bitwise_or(mask, m)
            if self.roi_frac < 0.999:             # zero out everything outside the central ROI
                mask[:ry0, :] = 0; mask[ry1:, :] = 0
                mask[:, :rx0] = 0; mask[:, rx1:] = 0
            kernel = self.cv2.getStructuringElement(self.cv2.MORPH_ELLIPSE, (5, 5))
            mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
            mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)
            n_labels, _, stats, centroids = self.cv2.connectedComponentsWithStats(mask)
            best_idx = -1
            best_area = self.min_blob_area
            for i in range(1, n_labels):
                area = stats[i, self.cv2.CC_STAT_AREA]
                if area > best_area:
                    best_area = area
                    best_idx = i
            accepted = False
            if best_idx >= 0:
                cx, cy = centroids[best_idx]
                xyz = self._pixel_to_world(cx, cy)
                xyz[:2] += self.pos_offset
                # World-frame sanity gate: reject back-projections outside the reachable table box
                # (a phantom skin blob teleports the world xy ~60cm outside this region).
                if bx0 <= xyz[0] <= bx1 and by0 <= xyz[1] <= by1:
                    out[ci] = xyz
                    self.last_positions[ci] = xyz
                    detected[ci] = True
                    accepted = True
                    debug.append((color, (float(cx), float(cy)), xyz.copy(), True))
            if not accepted:
                out[ci] = self.last_positions[ci]
                debug.append((color, None, self.last_positions[ci].copy(), False))
        self.last_detected = detected
        return (out, frame, debug) if return_debug else out

    def close(self):
        self.cap.release()


# =====================================================================
# Robot driver — lerobot SO101Follower (matches D:\soarm101 patterns)
# =====================================================================
class SO101Driver:
    def __init__(self, port: str, robot_id: str = "my_awesome_follower_arm"):
        try:
            from lerobot.robots.so101_follower.so101_follower import SO101Follower
            from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        except ImportError:
            raise RuntimeError(
                "Install lerobot >=0.2: pip install lerobot\n"
                "Project uses lerobot.robots.so101_follower API"
            )
        cfg = SO101FollowerConfig(
            port=port,
            id=robot_id,
            disable_torque_on_disconnect=True,
        )
        self.robot = SO101Follower(cfg)
        self.robot.connect(calibrate=False)
        print(f"Connected to SO-ARM101 at {port}")
        # Accumulated target qpos in radians (for pd_joint_target_delta_pos)
        self.target_qpos = REST_QPOS.copy()

    def read_qpos_rad(self) -> np.ndarray:
        """Returns (6,) current joint positions in radians."""
        # lerobot returns dict {"<name>.pos": value_in_degrees}
        obs = self.robot.get_observation()
        qpos_deg = np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
        # Apply elbow_flex offset (ManiSkill calibration convention)
        qpos_deg[2] -= ELBOW_FLEX_OFFSET_DEG
        return np.deg2rad(qpos_deg).astype(np.float32)

    def apply_action(self, action_6d: np.ndarray):
        """action[-1, 1] → delta × ACTION_RANGE → accumulate to target → send (degrees).
        Matches sim controller "pd_joint_target_delta_pos"."""
        action = np.clip(action_6d, -1.0, 1.0)
        delta = action * ACTION_RANGE
        self.target_qpos = self.target_qpos + delta
        # Clamp to reasonable joint limits (approximate; refine for your arm)
        self.target_qpos = np.clip(
            self.target_qpos,
            np.array([-2.5, -2.5, -2.5, -2.5, -2.5, 0.0]),
            np.array([ 2.5,  2.5,  2.5,  2.5,  2.5, 1.5]),
        )
        # Convert to degrees and build dict
        qpos_deg = np.rad2deg(self.target_qpos)
        qpos_deg[2] += ELBOW_FLEX_OFFSET_DEG  # reverse offset for send
        action_dict = {f"{name}.pos": float(qpos_deg[i]) for i, name in enumerate(JOINT_NAMES)}
        self.robot.send_action(action_dict)

    def reset_to_home(self, ramp_seconds: float = 2.0, control_hz: float = 30):
        """Smoothly move to REST_QPOS. Avoid snap motion that could damage arm."""
        print("Resetting to home pose (smooth)...")
        current = self.read_qpos_rad()
        target = REST_QPOS.copy()
        steps = int(ramp_seconds * control_hz)
        for i in range(steps):
            alpha = (i + 1) / steps
            interp = current + alpha * (target - current)
            self.target_qpos = interp
            qpos_deg = np.rad2deg(interp)
            qpos_deg[2] += ELBOW_FLEX_OFFSET_DEG
            action_dict = {f"{name}.pos": float(qpos_deg[i]) for i, name in enumerate(JOINT_NAMES)}
            self.robot.send_action(action_dict)
            time.sleep(1.0 / control_hz)
        time.sleep(0.5)
        self.target_qpos = target.copy()
        print("Reset done.")

    def close(self):
        try:
            self.robot.disconnect()
        except Exception:
            pass


# =====================================================================
# State vector — matches sim env layout (42 dim)
# =====================================================================
def compute_current_color_idx(cube_positions: np.ndarray, num_active: int = 5,
                              detected: np.ndarray = None) -> tuple[int, bool]:
    # Only the first num_active cubes are task cubes (n=2 -> red, orange). The rest are
    # distractors the policy was trained to ignore; never target them for placement.
    # A cube counts as placed ONLY if it is FRESHLY detected this frame — a stale/undetected
    # cube (e.g. lighting dropout) must NOT trigger a false success.
    for ci in range(num_active):
        xy = cube_positions[ci, :2]
        slot_xy = SLOT_XY[ci]
        in_slot_xy = np.linalg.norm(xy - slot_xy) < PLACEMENT_XY_TOL
        on_table = cube_positions[ci, 2] < PLACEMENT_Z_MAX
        is_det = True if detected is None else bool(detected[ci])
        if not (in_slot_xy and on_table and is_det):
            return ci, False
    return 0, True


def build_state_vector(qpos: np.ndarray, tcp_pos: np.ndarray, cube_positions: np.ndarray,
                       current_color_idx: int, all_done: bool) -> np.ndarray:
    state = np.zeros(42, dtype=np.float32)
    state[0:6] = qpos
    current_cube_xyz = cube_positions[current_color_idx]
    state[6:9] = current_cube_xyz - tcp_pos
    state[9:24] = cube_positions.reshape(-1)
    if not all_done:
        state[24 + current_color_idx] = 1.0
    state[29:39] = SLOT_XY.reshape(-1)
    state[39:42] = tcp_pos
    return state


# =====================================================================
# Deploy loop
# =====================================================================
def deploy(agent, robot, tracker, fk, max_steps=350, fix_wrist_roll=True,
           control_hz=CONTROL_HZ, verbose=True, action_ema=0.0, action_scale=1.0,
           num_active=2, tcp_z_offset=0.0):
    period = 1.0 / control_hz
    robot.reset_to_home(ramp_seconds=2.0)
    _colors = ["red", "orange", "yellow", "green", "blue"]
    print(f"\n=== Deploy: sort {num_active} cube(s) [{', '.join(_colors[:num_active])}] -> slots; "
          f"{5 - num_active} distractor(s) [{', '.join(_colors[num_active:]) or 'none'}] ignored ===")
    print(f"=== {max_steps} steps @ {control_hz}Hz | ema={action_ema} scale={action_scale} "
          f"tcp_z_offset={tcp_z_offset} ===\n")
    prev_action = np.zeros(6, dtype=np.float32)
    for t in range(max_steps):
        loop_start = time.time()
        qpos = robot.read_qpos_rad()
        tcp_pos = fk.tcp_pos(qpos)
        tcp_pos[2] += tcp_z_offset  # manual grasp-height correction for any FK z mismatch
        cube_pos = tracker.get_cube_positions()
        current_idx, all_done = compute_current_color_idx(
            cube_pos, num_active=num_active, detected=tracker.last_detected)
        if all_done:
            print(f"\n*** ALL {num_active} ACTIVE CUBES PLACED at step {t}! ***")
            return True
        state = build_state_vector(qpos, tcp_pos, cube_pos, current_idx, all_done)
        with torch.no_grad():
            action = agent.get_action(
                torch.from_numpy(state).unsqueeze(0).float()
            ).squeeze(0).numpy()
        action = action * action_scale
        if action_ema > 0.0:  # low-pass smoothing to damp oscillation
            action = action_ema * prev_action + (1.0 - action_ema) * action
        prev_action = action.copy()
        if fix_wrist_roll:
            action[4] = 0.0
        action = np.clip(action, -1.0, 1.0)
        robot.apply_action(action)

        if verbose and t % 10 == 0:
            colors = ["R", "O", "Y", "G", "B"]
            det = "SEE" if tracker.last_detected[current_idx] else "MISS<<"  # current cube detected?
            print(f"step {t:3d}  cur={colors[current_idx]}[{det}]  "
                  f"tcp=({tcp_pos[0]:+.2f},{tcp_pos[1]:+.2f},{tcp_pos[2]:+.2f})  "
                  f"cube_{colors[current_idx]}=({cube_pos[current_idx][0]:+.2f},"
                  f"{cube_pos[current_idx][1]:+.2f},{cube_pos[current_idx][2]:+.2f})  "
                  f"act=[{','.join(f'{x:+.2f}' for x in action[:6])}]")

        elapsed = time.time() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)
    print(f"\nMax steps reached. Last current_idx={current_idx}")
    return False


# =====================================================================
# Calibration helpers
# =====================================================================
def calibrate_camera_intrinsic(camera_index=0, chessboard=(9, 6), square_size=0.025):
    import cv2
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    obj_template = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    obj_template[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    obj_template *= square_size
    obj_list, img_list = [], []
    print(f"Show {chessboard} chessboard at various angles. SPACE to capture, ESC to finish (≥10 views).")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, chessboard, None)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, chessboard, corners, found)
        cv2.putText(disp, f"captured: {len(obj_list)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera Intrinsic Calibration", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break
        if k == 32 and found:
            obj_list.append(obj_template)
            img_list.append(corners)
            print(f"  captured {len(obj_list)}")
    cap.release()
    cv2.destroyAllWindows()
    if len(obj_list) < 5:
        raise RuntimeError("Need ≥5 views")
    ret, K, dist, _, _ = cv2.calibrateCamera(
        obj_list, img_list, gray.shape[::-1], None, None,
    )
    print(f"K =\n{K}\ndist = {dist.flatten()}")
    np.savez("calib_intrinsic.npz", K=K, dist=dist)
    print("Saved calib_intrinsic.npz")


def calibrate_extrinsic(camera_index=0, chessboard=(9, 6), square_size=0.025, board_z=0.0,
                        from_shoulder_pan=True):
    """Recover full T_robot_cam via solvePnP from ONE chessboard photo + 3 measured corners.

    Lay the 9x6 chessboard FLAT in the workspace (flat on the table, inside camera view).
    The live preview highlights 3 inner corners A(red)/B(blue)/C(green). You measure each
    one's robot-frame X,Y with a ruler. solvePnP then recovers the camera pose in the exact
    OpenCV optical convention _pixel_to_world uses — automatically absorbing any camera
    mounting tilt. No angle measurement needed.

    A=(0,0), B=far end of the 9-corner axis, C=far end of the 6-corner axis. objectPoints
    in robot frame are built by affine interpolation between A,B,C, so board in-plane
    rotation/scale is handled exactly from the 3 measurements.
    """
    import cv2
    nC, nR = chessboard            # 9 cols, 6 rows of inner corners
    iA, iB, iC = 0, nC - 1, (nR - 1) * nC   # indices of A(0,0), B(8,0), C(0,5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if not os.path.exists("calib_intrinsic.npz"):
        raise FileNotFoundError("calib_intrinsic.npz not found — run --calibrate_camera first.")
    d = np.load("calib_intrinsic.npz")
    K, dist = d["K"], d["dist"]

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    print("Lay the chessboard FLAT in the workspace. SPACE to capture (when all corners found), ESC to abort.")
    corners, disp = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, c = cv2.findChessboardCorners(
            gray, chessboard,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, chessboard, c, found)
            pA = tuple(c[iA, 0].astype(int))
            pB = tuple(c[iB, 0].astype(int))
            pC = tuple(c[iC, 0].astype(int))
            cv2.arrowedLine(disp, pA, pB, (255, 255, 0), 2, tipLength=0.04)  # i-axis A->B
            cv2.arrowedLine(disp, pA, pC, (0, 255, 255), 2, tipLength=0.04)  # j-axis A->C
            cv2.circle(disp, pA, 9, (0, 0, 255), -1)    # A red
            cv2.circle(disp, pB, 9, (255, 0, 0), -1)    # B blue
            cv2.circle(disp, pC, 9, (0, 200, 0), -1)    # C green
            for lbl, p, col in [("A", pA, (0, 0, 255)), ("B", pB, (255, 0, 0)), ("C", pC, (0, 200, 0))]:
                cv2.putText(disp, lbl, (p[0] + 12, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.putText(disp, "FOUND - SPACE to capture" if found else "searching for board...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("Extrinsic Calibration", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cap.release(); cv2.destroyAllWindows(); print("Aborted."); return
        if k == 32 and found:
            corners = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), criteria)
            break
    cap.release(); cv2.destroyAllWindows()

    cv2.imwrite("extrinsic_capture.png", disp)
    print("Saved extrinsic_capture.png — open it to see which corners are A(red)/B(blue)/C(green).")

    origin_name = ("SHOULDER-PAN axis (the J0 left/right rotation axis, directly below it)"
                   if from_shoulder_pan else
                   "Base-link world origin (4.52cm BEHIND the shoulder-pan axis)")
    print(f"\nMeasure each highlighted corner's position from the {origin_name}.")
    print("Robot frame: +X forward (toward slots), +Y left.  Units = METERS.\n")
    Ax = float(input("  A (red)   robot X (m): ")); Ay = float(input("  A (red)   robot Y (m): "))
    Bx = float(input("  B (blue)  robot X (m): ")); By = float(input("  B (blue)  robot Y (m): "))
    Cx = float(input("  C (green) robot X (m): ")); Cy = float(input("  C (green) robot Y (m): "))
    A = np.array([Ax, Ay]); B = np.array([Bx, By]); C = np.array([Cx, Cy])
    if from_shoulder_pan:
        # Convert shoulder-pan-referenced measurements into the sim/FK WORLD frame.
        A = A + SHOULDER_PAN_WORLD_XY
        B = B + SHOULDER_PAN_WORLD_XY
        C = C + SHOULDER_PAN_WORLD_XY
        print(f"[frame] measured from shoulder-pan axis -> added world offset "
              f"{SHOULDER_PAN_WORLD_XY.tolist()} m so calib lives in the sim/FK world frame.")

    meas_i = np.linalg.norm(B - A) / (nC - 1)
    meas_j = np.linalg.norm(C - A) / (nR - 1)
    print(f"\nMeasured square size: i-axis={meas_i*1000:.1f}mm/sq, j-axis={meas_j*1000:.1f}mm/sq "
          f"(nominal {square_size*1000:.0f}mm)")
    if abs(meas_i - square_size) > 0.01 or abs(meas_j - square_size) > 0.01:
        print("  WARNING: measured spacing differs >10mm from nominal — re-check which physical")
        print("           corners are A/B/C and your ruler readings before trusting the result.")

    # objectPoints in robot frame via affine interpolation between A, B, C
    obj = np.zeros((nC * nR, 3), np.float32)
    for k in range(nC * nR):
        ci, ri = k % nC, k // nC
        xy = A + (ci / (nC - 1)) * (B - A) + (ri / (nR - 1)) * (C - A)
        obj[k, :2] = xy
        obj[k, 2] = board_z
    img = corners.reshape(-1, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    reproj_err = np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(axis=1)).mean()
    print(f"\nsolvePnP reprojection error: {reproj_err:.2f} px  (good < 1.0, acceptable < 2.0)")

    R_cr, _ = cv2.Rodrigues(rvec)        # rotation: robot -> cam
    t_cr = tvec.reshape(3)
    R_rc = R_cr.T                        # invert: cam -> robot
    t_rc = -R_rc @ t_cr
    T_robot_cam = np.eye(4)
    T_robot_cam[:3, :3] = R_rc
    T_robot_cam[:3, 3] = t_rc
    print(f"\nRecovered camera position in robot frame: "
          f"X={t_rc[0]:+.3f}  Y={t_rc[1]:+.3f}  Z={t_rc[2]:+.3f} (m)")
    print("  ^ sanity-check this against where the camera physically sits (esp. Z = height).")
    print(f"T_robot_cam =\n{np.array2string(T_robot_cam, precision=4, suppress_small=True)}")

    # Round-trip: pixel -> world using the SAME math as deploy's _pixel_to_world
    print("\nRound-trip check (pixel -> world via recovered calib; should match your measurements):")
    max_err = 0.0
    for name, idx, meas in [("A", iA, A), ("B", iB, B), ("C", iC, C)]:
        u, v = img[idx]
        und = cv2.undistortPoints(np.array([[[u, v]]], np.float32), K, dist).reshape(2)
        ray_world = R_rc @ np.array([und[0], und[1], 1.0])
        s = (board_z - t_rc[2]) / ray_world[2]
        w = (t_rc + s * ray_world)[:2]
        err = np.linalg.norm(w - meas) * 1000
        max_err = max(max_err, err)
        print(f"  {name}: recovered=({w[0]:+.3f},{w[1]:+.3f})  measured=({meas[0]:+.3f},{meas[1]:+.3f})  err={err:.1f}mm")
    if reproj_err > 2.0 or max_err > 10.0:
        print("\n  NOTE: errors are high. Re-check ruler readings / board flatness / intrinsic quality.")

    np.savez("calib.npz", K=K, dist=dist, T_robot_cam=T_robot_cam)
    print("\nSaved calib.npz with K, dist, T_robot_cam. Ready for deploy.")


def probe_hsv(camera_index=0):
    """Hover/click a cube to read its true median HSV. Use the printed H values to
    fix CubeTracker.COLOR_HSV_RANGES so each color is cleanly separated."""
    import cv2
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    st = {"pos": None, "do_print": False}

    def on_mouse(event, x, y, flags, param):
        st["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            st["do_print"] = True

    win = "HSV probe (hover=read, click=print, ESC=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Hover a cube to see its HSV; CLICK to print it. ESC to quit.")
    print("Report me the H (hue) of each colored cube so I can fix the ranges.\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        disp = frame.copy()
        if st["pos"] is not None:
            x, y = st["pos"]; r = 4
            h, w = hsv.shape[:2]
            patch = hsv[max(0, y - r):min(h, y + r), max(0, x - r):min(w, x + r)].reshape(-1, 3)
            if len(patch):
                med = np.median(patch, axis=0)
                txt = f"H={med[0]:.0f} S={med[1]:.0f} V={med[2]:.0f}"
                cv2.circle(disp, (x, y), r, (255, 255, 255), 2)
                cv2.putText(disp, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if st["do_print"]:
                    print(f"  pixel ({x},{y})  median {txt}")
                    st["do_print"] = False
        cv2.imshow(win, disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def tune_hsv(camera_index=0):
    import cv2
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cv2.namedWindow("Tune HSV")
    for n, v in [("H_lo", 0), ("H_hi", 180), ("S_lo", 100), ("S_hi", 255), ("V_lo", 100), ("V_hi", 255)]:
        cv2.createTrackbar(n, "Tune HSV", v, 255 if n[0] != "H" else 180, lambda x: None)
    print("Adjust sliders, ESC to exit")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array([cv2.getTrackbarPos(c, "Tune HSV") for c in ["H_lo", "S_lo", "V_lo"]])
        hi = np.array([cv2.getTrackbarPos(c, "Tune HSV") for c in ["H_hi", "S_hi", "V_hi"]])
        mask = cv2.inRange(hsv, lo, hi)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Tune HSV", np.hstack([frame, masked]))
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()


def read_qpos_mode(port, robot_id):
    """Disable torque; continuously print the 6 joint angles (deg). Move the arm by hand to a
    desired pose and read the angles — e.g. to pick the wrist_roll lock value for --wrist_roll_deg."""
    robot = SO101Driver(port=port, robot_id=robot_id)
    try:
        robot.robot.bus.disable_torque()
        print("\nTORQUE OFF — move the arm by hand. Joint angles (deg). Ctrl-C to stop.\n")
        while True:
            deg = np.rad2deg(robot.read_qpos_rad())
            print("\r " + "  ".join(f"{n}={d:+6.1f}" for n, d in zip(JOINT_NAMES, deg)) + "    ",
                  end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        try:
            robot.robot.bus.enable_torque()
        except Exception:
            pass
        robot.close()


def reach_test(port, robot_id):
    """Disable torque; you hand-pose the arm to its max forward table-height reach.
    Reads qpos -> FK -> world TCP x, tracks the max, compares to slot x=0.40.
    Tells you definitively whether the real arm can place cubes at the slots."""
    fk = SO100ForwardKinematics()
    robot = SO101Driver(port=port, robot_id=robot_id)
    max_x, max_pose = -1e9, None
    try:
        robot.robot.bus.disable_torque()
        print("\n" + "=" * 64)
        print("TORQUE DISABLED — move the arm BY HAND.")
        print("Slowly extend the gripper FORWARD, keeping it near table height (z<0.10),")
        print("as far as it could still grasp/release a cube. Watch 'world x'.")
        print("Press Ctrl-C when you've reached the farthest pose.")
        print("=" * 64 + "\n")
        while True:
            q = robot.read_qpos_rad()
            tcp = fk.tcp_pos(q)
            if tcp[2] < 0.10 and tcp[0] > max_x:
                max_x, max_pose = float(tcp[0]), q.copy()
            print(f"\r world tcp x={tcp[0]:+.3f} y={tcp[1]:+.3f} z={tcp[2]:+.3f} | "
                  f"max@table={max_x:.3f} | slot=0.40 (need>=0.375)   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\n" + "=" * 64)
        print(f"Max forward reach at table height: world x = {max_x:.3f} m "
              f"(= {max_x - 0.0452:.3f} m from shoulder-pan axis)")
        print("slot x = 0.40 ; placement needs tcp x >= 0.375 (with 2.5cm tolerance)")
        if max_x >= 0.40:
            print("VERDICT: reaches slots comfortably — no reach-related retrain needed.")
        elif max_x >= 0.375:
            print("VERDICT: MARGINAL — only the tolerance edge; placement will be unreliable.")
        else:
            print(f"VERDICT: SHORT by {(0.40 - max_x) * 100:.1f} cm. "
                  f"Retrain with SLOT_XY x pulled in to ~{max_x - 0.03:.2f}.")
        print("=" * 64)
        try:
            robot.robot.bus.enable_torque()
        except Exception:
            pass
        robot.close()


def test_perception(camera_index, calibration, roi_frac=DETECT_ROI_FRAC):
    """Live HSV cube detection -> WORLD coords (no arm movement). Place a cube at a
    ruler-measured spot and confirm the reported world (x,y) matches. Validates the
    whole calib + origin-offset + HSV chain BEFORE letting the arm move to those coords.
    The green box = detection ROI (edge clutter outside it is ignored)."""
    import cv2
    if not os.path.exists(calibration):
        raise FileNotFoundError(f"{calibration} not found — run --calibrate_extrinsic first.")
    c = np.load(calibration)
    dist = c["dist"] if "dist" in c else None
    tracker = CubeTracker(camera_index=camera_index, K=c["K"], dist=dist,
                          T_robot_cam=c["T_robot_cam"],
                          pos_offset=(c["pos_offset"] if "pos_offset" in c.files else None),
                          roi_frac=roi_frac)
    bgr = {"red": (0, 0, 255), "orange": (0, 128, 255), "yellow": (0, 255, 255),
           "green": (0, 200, 0), "blue": (255, 0, 0)}
    print("Live cube WORLD positions (meters, robot frame). ESC in the window to quit.")
    print("Place a cube at a ruler-measured spot; check its (x,y) below matches.\n")
    try:
        while True:
            out, frame, dbg = tracker.get_cube_positions(return_debug=True)
            if frame is None:
                continue
            H, W = frame.shape[:2]                # draw the detection ROI box (green)
            rx0, rx1 = int(W * (1 - roi_frac) / 2), int(W * (1 + roi_frac) / 2)
            ry0, ry1 = int(H * (1 - roi_frac) / 2), int(H * (1 + roi_frac) / 2)
            cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 255, 0), 2)
            parts = []
            for color, px, xyz, found in dbg:
                k = color[0].upper()
                parts.append(f"{k}:({xyz[0]:+.3f},{xyz[1]:+.3f})" if found else f"{k}:--")
                if found and px is not None:
                    p = (int(px[0]), int(px[1]))
                    cv2.circle(frame, p, 6, bgr[color], -1)
                    cv2.putText(frame, f"{xyz[0]:.2f},{xyz[1]:.2f}", (p[0] + 8, p[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr[color], 1)
            print("\r " + "   ".join(parts) + "      ", end="", flush=True)
            cv2.imshow("Perception test — world xy (ESC quits)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        print()
        tracker.close()
        cv2.destroyAllWindows()


# =====================================================================
# Main
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--robot_port", default="COM4")
    p.add_argument("--robot_id", default="my_awesome_follower_arm")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--calibration", default="calib.npz")
    p.add_argument("--roi_frac", type=float, default=DETECT_ROI_FRAC,
                   help="central image fraction kept for detection (1.0=whole frame; lower drops edges)")
    p.add_argument("--max_steps", type=int, default=350)
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--num_active", type=int, default=2,
                   help="# of task cubes (lowest indices) to sort; rest ignored. DEFAULT 2 (red+orange) for the "
                        "current n=2 experiment. Pass --num_active=5 for the full 5-cube task.")
    p.add_argument("--slot_x", type=float, default=None,
                   help="Override slot-row x (m). Use 0.40 to deploy the OLD v16 5-cube policy "
                        "(trained at slots x=0.40); default keeps the env's current SLOT_XY (0.28).")
    p.add_argument("--control_hz", type=float, default=CONTROL_HZ)
    p.add_argument("--no_wrist_roll_fix", action="store_true")
    p.add_argument("--wrist_roll_deg", type=float, default=None,
                   help="Locked wrist_roll angle in degrees (default 90). Try to fix gripper orientation.")
    p.add_argument("--action_ema", type=float, default=0.0,
                   help="EMA smoothing 0..1 to damp action oscillation (try 0.5-0.8)")
    p.add_argument("--action_scale", type=float, default=1.0,
                   help="Scale policy actions (try 0.5 to slow/soften motion)")
    p.add_argument("--tcp_z_offset", type=float, default=0.0,
                   help="Manual FK grasp-height correction (m) added to tcp z. Tune empirically if real "
                        "grasps are systematically too high/low (e.g. -0.01 if it grasps ~1cm too high).")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--calibrate_camera", action="store_true")
    p.add_argument("--calibrate_extrinsic", action="store_true",
                   help="Recover T_robot_cam via solvePnP (board flat in workspace)")
    p.add_argument("--square_size", type=float, default=0.025,
                   help="Chessboard square size in meters (default 25mm)")
    p.add_argument("--board_z", type=float, default=0.0,
                   help="Robot-frame Z of the board surface (m); table top = 0.0")
    p.add_argument("--from_base", action="store_true",
                   help="A/B/C measured from the URDF Base-link world origin (skip the "
                        "+4.52cm shoulder-pan->world offset). Default assumes shoulder-pan axis.")
    p.add_argument("--tune_hsv", action="store_true")
    p.add_argument("--probe_hsv", action="store_true",
                   help="Hover/click cubes to read their true HSV (fix color ranges)")
    p.add_argument("--test_fk", action="store_true", help="Test FK only (no hardware)")
    p.add_argument("--reach_test", action="store_true",
                   help="Disable torque; hand-pose arm to measure max forward reach via FK")
    p.add_argument("--read_qpos", action="store_true",
                   help="Disable torque; live-print joint angles (deg) to pick a wrist_roll lock value")
    p.add_argument("--test_perception", action="store_true",
                   help="Live cube world-position readout (no arm) to validate calib + HSV")
    args = p.parse_args()

    if args.calibrate_camera:
        calibrate_camera_intrinsic(args.camera_index); return
    if args.calibrate_extrinsic:
        calibrate_extrinsic(args.camera_index, square_size=args.square_size,
                            board_z=args.board_z, from_shoulder_pan=not args.from_base); return
    if args.tune_hsv:
        tune_hsv(args.camera_index); return
    if args.test_perception:
        test_perception(args.camera_index, args.calibration, roi_frac=args.roi_frac); return
    if args.probe_hsv:
        probe_hsv(args.camera_index); return
    if args.test_fk:
        fk = SO100ForwardKinematics()
        print(f"\nTCP at rest qpos {REST_QPOS}: {fk.tcp_pos(REST_QPOS)}")
        print(f"TCP at zero qpos: {fk.tcp_pos(np.zeros(6))}")
        return
    if args.reach_test:
        reach_test(args.robot_port, args.robot_id); return
    if args.read_qpos:
        read_qpos_mode(args.robot_port, args.robot_id); return

    if args.ckpt is None:
        print("--ckpt required (or use --calibrate_camera / --tune_hsv / --test_fk)"); return

    if args.slot_x is not None:
        SLOT_XY[:, 0] = args.slot_x  # match the policy's trained slot row (v16 -> 0.40)
        print(f"Slot-row x overridden to {args.slot_x} (slots: {SLOT_XY.tolist()})")
    if args.wrist_roll_deg is not None:
        REST_QPOS[4] = np.deg2rad(args.wrist_roll_deg)
        print(f"Locked wrist_roll set to {args.wrist_roll_deg} deg ({REST_QPOS[4]:.3f} rad)")

    # Load policy
    agent = StateAgent()
    sd = torch.load(args.ckpt, map_location="cpu")
    agent.load_state_dict(sd)
    agent.eval()
    print(f"Loaded policy: {args.ckpt}")
    print(f"  actor_logstd: {sd['actor_logstd'].squeeze().tolist()}")

    # FK (always needed)
    fk = SO100ForwardKinematics()

    if args.dry_run:
        print(f"--dry_run: policy + FK loaded OK. TCP@rest = {fk.tcp_pos(REST_QPOS)}")
        return

    # Load calibration
    if not os.path.exists(args.calibration):
        raise FileNotFoundError(
            f"{args.calibration} not found. Run --calibrate_camera then build_T_robot_cam.py"
        )
    calib = np.load(args.calibration)
    if "T_robot_cam" not in calib:
        raise RuntimeError(f"{args.calibration} missing T_robot_cam. Run build_T_robot_cam.py")
    K, T_robot_cam = calib["K"], calib["T_robot_cam"]
    dist = calib["dist"] if "dist" in calib else None
    print(f"Calibration loaded: K shape {K.shape}, T_robot_cam shape {T_robot_cam.shape}, "
          f"dist {'present' if dist is not None else 'MISSING (no undistort)'}")

    # Hardware
    pos_offset = calib["pos_offset"] if "pos_offset" in calib.files else None
    if pos_offset is not None:
        print(f"  pos_offset correction: {np.asarray(pos_offset).tolist()}")
    tracker = CubeTracker(camera_index=args.camera_index, K=K, dist=dist,
                          T_robot_cam=T_robot_cam, pos_offset=pos_offset, roi_frac=args.roi_frac)
    robot = SO101Driver(port=args.robot_port, robot_id=args.robot_id)

    try:
        successes = 0
        for ep in range(args.n_episodes):
            print(f"\n========== Episode {ep + 1}/{args.n_episodes} ==========")
            input(f"Position cubes in workspace (first {args.num_active} colors R,O,Y,G,B are the "
                  f"task cubes). Press ENTER to start...")
            ok = deploy(
                agent, robot, tracker, fk,
                max_steps=args.max_steps,
                fix_wrist_roll=not args.no_wrist_roll_fix,
                control_hz=args.control_hz,
                action_ema=args.action_ema,
                action_scale=args.action_scale,
                num_active=args.num_active,
                tcp_z_offset=args.tcp_z_offset,
            )
            if ok:
                successes += 1
        print(f"\n=== FINAL: {successes}/{args.n_episodes} successful ===")
    finally:
        robot.close()
        tracker.close()


if __name__ == "__main__":
    main()
