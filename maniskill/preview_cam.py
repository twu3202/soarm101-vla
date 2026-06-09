"""Preview camera frames to identify which index is which physical camera.

Usage:
    python preview_cam.py 0     # show camera 0 live (press ESC to close)
    python preview_cam.py 1     # show camera 1
"""
import sys
import cv2

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print(f"Opening camera index {idx} (DSHOW)... press ESC to close")
cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Failed to open camera"); sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame"); break
    cv2.putText(frame, f"camera idx={idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(f"Camera {idx}", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
