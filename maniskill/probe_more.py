"""Probe cameras 0-8 with DSHOW (skips MSMF phantom devices)."""
import cv2
for idx in range(0, 9):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    shape = "-"
    if ok:
        for _ in range(3):  # warm-up grabs
            ret, frame = cap.read()
        ret, frame = cap.read()
        if ret and frame is not None:
            shape = f"{frame.shape}"
        else:
            shape = "open but no frame"
    cap.release()
    print(f"idx {idx}: opened={ok}  frame={shape}")
