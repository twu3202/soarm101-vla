"""Enumerate connected cameras (Windows, DSHOW) to identify which index is the TOP
camera (overhead) vs the WRIST camera (mounted on the gripper).

Saves one JPG snapshot per working camera to ./cam_snaps/ so you can eyeball them and
tell which index is which. We need BOTH indices to record the 2-camera dataset.

Run:  python enum_cameras.py
Then open vla_sort\\cam_snaps\\cam0.jpg, cam1.jpg, ... and note:
  - which index shows the overhead / top-down view      -> that's "top"
  - which index shows the close-up gripper/finger view  -> that's "wrist"
"""
import os
import cv2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cam_snaps")
os.makedirs(OUT, exist_ok=True)

print("Probing camera indices 0..7 (CAP_DSHOW)...\n")
found = []
for idx in range(8):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        cap.release()
        continue
    ok, frame = False, None
    for _ in range(10):  # warm up / flush
        ok, frame = cap.read()
    if ok and frame is not None:
        h, w = frame.shape[:2]
        m, s = float(frame.mean()), float(frame.std())
        path = os.path.join(OUT, f"cam{idx}.jpg")
        cv2.imwrite(path, frame)
        live = "LIVE" if s > 5 else "DEAD/blank"
        print(f"  index {idx}: {w}x{h}  mean={m:.1f} std={s:.1f}  [{live}]  -> {path}")
        found.append(idx)
    else:
        print(f"  index {idx}: opened but read failed")
    cap.release()

print(f"\nWorking cameras: {found}")
print(f"Open the JPGs in {OUT}\\ and tell me which index is TOP (overhead) and which is WRIST (gripper).")
