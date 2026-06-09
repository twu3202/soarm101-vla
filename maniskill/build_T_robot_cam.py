"""Helper: build T_robot_cam (4x4) from manually-measured camera pose.

Run this once after physically mounting the camera. Asks for camera position
+ orientation relative to robot base, computes the 4x4 transform, merges with
calib_intrinsic.npz and saves as calib.npz.

Robot frame: x = forward (toward slots), y = left, z = up. Origin at base.

Usage: python build_T_robot_cam.py
"""
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

print("=== T_robot_cam builder ===\n")
print("Measure your camera's position relative to robot base origin.")
print("Robot frame convention: +x forward, +y left, +z up\n")

x = float(input("Camera position X (m, e.g. 0.30 if camera is 30cm in front of base): "))
y = float(input("Camera position Y (m): "))
z = float(input("Camera position Z (m, e.g. 0.45 if camera is 45cm above base): "))

print("\nCamera orientation: how is the camera lens pointed?")
print("  For TOP-DOWN camera looking straight down at workspace:")
print("    yaw=0, pitch=-90 (camera Z axis points down)")
print("  For SIDE camera looking horizontally toward slots:")
print("    yaw=0, pitch=0  (camera Z points forward = +x of robot)")

yaw_deg = float(input("yaw (deg): "))
pitch_deg = float(input("pitch (deg): "))
roll_deg = float(input("roll (deg, usually 0): "))

# Convert to rotation matrix (Z * Y * X order = yaw, pitch, roll)
rot = R.from_euler('zyx', [yaw_deg, pitch_deg, roll_deg], degrees=True)
R_robot_cam = rot.as_matrix()
T_robot_cam = np.eye(4)
T_robot_cam[:3, :3] = R_robot_cam
T_robot_cam[:3, 3] = [x, y, z]

print(f"\nT_robot_cam =\n{T_robot_cam}")

# Merge with intrinsic
intrinsic_path = "calib_intrinsic.npz"
if not os.path.exists(intrinsic_path):
    print(f"\nWARNING: {intrinsic_path} not found. Run --calibrate_camera first.")
    K = np.eye(3)
    dist = np.zeros(5)
else:
    d = np.load(intrinsic_path)
    K, dist = d["K"], d["dist"]
    print(f"Loaded K from {intrinsic_path}")

np.savez("calib.npz", K=K, dist=dist, T_robot_cam=T_robot_cam)
print("\nSaved calib.npz with K, dist, T_robot_cam")
print("Ready for: python deploy_real.py --ckpt=... --calibration=calib.npz")
