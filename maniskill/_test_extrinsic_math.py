"""Synthetic validation of the solvePnP extrinsic-calibration math used in
deploy_real.calibrate_extrinsic. Simulates a tilted top-down camera viewing a
chessboard at a known robot-frame pose, then checks that solvePnP + inversion
recovers T_robot_cam and that the pixel->world round-trip matches ground truth.
No hardware/camera needed.
"""
import numpy as np
import cv2

np.set_printoptions(precision=4, suppress=True)

# ---- Ground-truth camera intrinsics (close to the real calib) ----
K = np.array([[437.3, 0, 374.9],
              [0, 434.1, 314.6],
              [0, 0, 1.0]])
dist = np.array([-0.389, 0.145, -0.0137, -0.0211, -0.030])

# ---- Ground-truth camera pose in robot frame (top-down, slightly tilted) ----
# Camera ~45cm above, 27cm in front, tilted 7 deg off vertical, rotated 10 deg in-plane.
from scipy.spatial.transform import Rotation as R
# Build a rotation whose optical axis (+z) points roughly down (-z robot):
#   start from "looking straight down" then add small tilt + in-plane spin.
R_down = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1.0]])  # cam +z -> -z robot, cam +x -> +x, cam +y -> -y
tilt = R.from_euler("xyz", [7, -4, 10], degrees=True).as_matrix()
R_rc_true = tilt @ R_down                      # cam -> robot
t_rc_true = np.array([0.27, 0.02, 0.46])       # camera position in robot frame
T_true = np.eye(4); T_true[:3, :3] = R_rc_true; T_true[:3, 3] = t_rc_true

# ---- Chessboard: 9x6 inner corners, 25mm, lying flat at z=0, placed in workspace ----
nC, nR, sq, board_z = 9, 6, 0.025, 0.0
# Board origin at (0.18, -0.10) robot frame, rotated 12 deg in-plane (to stress the affine recovery)
theta = np.deg2rad(12.0)
ax_i = np.array([np.cos(theta), np.sin(theta)]) * sq
ax_j = np.array([-np.sin(theta), np.cos(theta)]) * sq
origin = np.array([0.18, -0.10])
obj_true = np.zeros((nC * nR, 3), np.float32)
for k in range(nC * nR):
    ci, ri = k % nC, k // nC
    obj_true[k, :2] = origin + ci * ax_i + ri * ax_j
    obj_true[k, 2] = board_z

# ---- Project to image: world -> cam -> pixel ----
R_cr_true = R_rc_true.T
t_cr_true = -R_cr_true @ t_rc_true
rvec_true, _ = cv2.Rodrigues(R_cr_true)
img, _ = cv2.projectPoints(obj_true, rvec_true, t_cr_true.reshape(3, 1), K, dist)
img = img.reshape(-1, 2).astype(np.float32)
# add sub-pixel noise to mimic cornerSubPix imperfection
rng = np.random.RandomState(0)
NOISE = float(__import__("os").environ.get("NOISE", "0.2"))
img += rng.normal(0, NOISE, img.shape).astype(np.float32)

# ---- What the user measures: robot-frame XY of corners A(0), B(8), C(45) ----
iA, iB, iC = 0, nC - 1, (nR - 1) * nC
A = obj_true[iA, :2].astype(float)
B = obj_true[iB, :2].astype(float)
C = obj_true[iC, :2].astype(float)
# add ruler measurement noise (RULER meters std)
RULER = float(__import__("os").environ.get("RULER", "0.002"))
A += rng.normal(0, RULER, 2); B += rng.normal(0, RULER, 2); C += rng.normal(0, RULER, 2)

# ===== Replicate calibrate_extrinsic logic =====
obj = np.zeros((nC * nR, 3), np.float32)
for k in range(nC * nR):
    ci, ri = k % nC, k // nC
    xy = A + (ci / (nC - 1)) * (B - A) + (ri / (nR - 1)) * (C - A)
    obj[k, :2] = xy
    obj[k, 2] = board_z

ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
reproj_err = np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(axis=1)).mean()
R_cr, _ = cv2.Rodrigues(rvec)
t_cr = tvec.reshape(3)
R_rc = R_cr.T
t_rc = -R_rc @ t_cr

print(f"solvePnP reprojection error: {reproj_err:.3f} px")
print(f"recovered cam pos : {t_rc}")
print(f"true      cam pos : {t_rc_true}")
print(f"cam pos error     : {np.linalg.norm(t_rc - t_rc_true)*1000:.2f} mm")

# ---- Round-trip: project ALL corners pixel->world, compare to TRUE robot positions ----
errs = []
for k in range(nC * nR):
    u, v = img[k]
    und = cv2.undistortPoints(np.array([[[u, v]]], np.float32), K, dist).reshape(2)
    ray_world = R_rc @ np.array([und[0], und[1], 1.0])
    s = (board_z - t_rc[2]) / ray_world[2]
    w = (t_rc + s * ray_world)[:2]
    errs.append(np.linalg.norm(w - obj_true[k, :2]) * 1000)
errs = np.array(errs)
print(f"\npixel->world error over all 54 corners vs GROUND TRUTH:")
print(f"  mean={errs.mean():.2f}mm  max={errs.max():.2f}mm")
print("\nPASS" if errs.mean() < 5 and reproj_err < 1.0 else "\nCHECK (errors higher than expected)")
