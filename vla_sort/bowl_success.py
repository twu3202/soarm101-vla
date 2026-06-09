"""Sparse success / reward detector for the green-cube-into-bowl task, from the TOP camera.

success == the GREEN cube's centroid lies inside the BLUE bowl disk.

This is the reward function for the RL ladder (see RL_PLAN.md): rung-1 iterative-SFT keeps episodes
where cube_in_bowl()==True; rung-2 AWR uses it as the 0/1 reward.

Self-test (validates the HSV thresholds against the real demos — every demo ENDS in success and
STARTS out of the bowl):
    "C:\\Users\\asus\\miniconda3\\envs\\lerobot\\python.exe" vla_sort/bowl_success.py
"""
import numpy as np
import cv2

# --- HSV thresholds (OpenCV H:0-180). Tuned from green-bowl demo frames. ---
BLUE_LO = np.array([90, 35, 70], np.uint8)
BLUE_HI = np.array([130, 255, 255], np.uint8)
GREEN_LO = np.array([35, 70, 50], np.uint8)
GREEN_HI = np.array([85, 255, 255], np.uint8)

MIN_BOWL_AREA = 1500   # px^2; the bowl is a big blob, rejects the thin blue table-arc
MIN_GREEN_AREA = 100   # px^2; rejects green speckle noise
INSIDE_MARGIN = 0.90   # cube center must be within 90% of the bowl radius


def _largest_blob(mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_a = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= min_area and a > best_a:
            best, best_a = c, a
    return best, best_a


def cube_in_bowl(rgb):
    """rgb: HxWx3 uint8 (RGB). Returns (success: bool, info: dict)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    bowl_c, bowl_a = _largest_blob(blue, MIN_BOWL_AREA)
    green_c, green_a = _largest_blob(green, MIN_GREEN_AREA)
    info = {"bowl_found": bowl_c is not None, "green_found": green_c is not None,
            "bowl_area": bowl_a, "green_area": green_a}
    if bowl_c is None or green_c is None:
        info["success"] = False
        return False, info

    (bx, by), br = cv2.minEnclosingCircle(bowl_c)
    M = cv2.moments(green_c)
    gx, gy = M["m10"] / (M["m00"] + 1e-6), M["m01"] / (M["m00"] + 1e-6)
    dist = float(np.hypot(gx - bx, gy - by))
    success = dist < br * INSIDE_MARGIN
    info.update({"bowl_xy": (round(bx, 1), round(by, 1)), "bowl_r": round(br, 1),
                 "green_xy": (round(gx, 1), round(gy, 1)), "dist": round(dist, 1),
                 "success": success})
    return success, info


def _selftest():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/so101_green_bowl")

    try:
        frm = ds.episode_data_index["from"].tolist()
        to = ds.episode_data_index["to"].tolist()
    except Exception:  # fallback: frame_index resets to 0 at each episode start
        fis = [int(ds[i]["frame_index"]) for i in range(len(ds))]
        frm = [i for i, f in enumerate(fis) if f == 0]
        to = frm[1:] + [len(ds)]

    def top_rgb(i):
        a = np.asarray(ds[i]["observation.images.top"])
        if a.dtype != np.uint8:
            a = (255 * a).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
        if a.ndim == 3 and a.shape[0] == 3:
            a = np.transpose(a, (1, 2, 0))
        return a

    n_ep = len(frm)
    start_succ = end_succ = 0
    print(f"{n_ep} episodes\n ep |        start (expect FAIL)         |          end (expect OK)")
    for k in range(n_ep):
        s_ok, s_i = cube_in_bowl(top_rgb(frm[k]))
        e_ok, e_i = cube_in_bowl(top_rgb(to[k] - 1))
        start_succ += s_ok
        end_succ += e_ok
        print(f" {k:2d} | succ={int(s_ok)} {str(s_i.get('green_xy')):>14} d={s_i.get('dist')}/"
              f"{s_i.get('bowl_r')} | succ={int(e_ok)} {str(e_i.get('green_xy')):>14} "
              f"d={e_i.get('dist')}/{e_i.get('bowl_r')}")
    print(f"\nSTART success rate: {start_succ}/{n_ep}  (want ~0)")
    print(f"END   success rate: {end_succ}/{n_ep}  (want ~{n_ep})")
    print("Detector is good if END is high and START is low.")


if __name__ == "__main__":
    _selftest()
