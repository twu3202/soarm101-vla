"""Deploy the trained pi0.5 SO-101 GREEN-CUBE-INTO-BOWL policy on the REAL arm.

Server runs serve_so101.sh (openpi websocket policy server, serving pi05_green_bowl_lora).
This client runs on the Windows arm machine. Each cycle: read SO-101 follower joint state (deg)
+ a LIVE top-camera frame + a LIVE wrist-camera frame, send to the server, execute the returned
absolute-joint-target chunk (deg).

Cameras (Windows): grabbed with cv2 + CAP_DSHOW and converted BGR->RGB to match the LeRobot
recording pipeline.  top = index 0 (overhead)   wrist = index 1 (gripper).

Setup:  scp -r twu@hy-trx50-ai-top.local:Projects/openpi/packages/openpi-client/src/openpi_client .
        pip install websockets msgpack opencv-python
Run (server reachable via the SSH tunnel on localhost:8000):
  python so101_vla_deploy.py --server_host=localhost --robot_port=COM4 --camera_index=0 --wrist_index=1
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find ./openpi_client
from openpi_client import websocket_client_policy  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server_host", default="localhost")
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument("--robot_port", default="COM4")
    p.add_argument("--robot_id", default="my_awesome_follower_arm")
    p.add_argument("--camera_index", type=int, default=0, help="TOP (overhead) camera cv2 index")
    p.add_argument("--wrist_index", type=int, default=1, help="WRIST (gripper) camera cv2 index")
    p.add_argument("--prompt", default="Put the green cube in the bowl")
    p.add_argument("--exec_horizon", type=int, default=5, help="actions to run before re-infer (tight loop = less drift)")
    p.add_argument("--smooth", type=float, default=1.0, help="EMA weight on new target for the 5 ARM joints (1.0=off; 0.5 halves jitter). Gripper is NOT smoothed.")
    p.add_argument("--hz", type=float, default=30.0, help="control rate; dataset is 30fps so 30 matches training")
    p.add_argument("--max_cycles", type=int, default=200)
    p.add_argument("--home", default="none", help="comma joint deg to reset to before inference; 'none' to skip")
    p.add_argument("--reset_secs", type=float, default=2.5)
    args = p.parse_args()

    import cv2

    # --- follower (STATE only; cameras handled separately via cv2 DSHOW) ---
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(port=args.robot_port, id=args.robot_id))
    robot.connect(calibrate=False)
    print(f"Follower connected at {args.robot_port}")

    # --- two cameras via cv2 + DSHOW (live frames on Windows); BGR->RGB to match training ---
    def open_cam(idx, name):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {name} camera index {idx}")
        for _ in range(8):
            cap.read()  # warm up / flush buffer
        return cap

    cap_top = open_cam(args.camera_index, "TOP")
    cap_wrist = open_cam(args.wrist_index, "WRIST")

    def grab_rgb(cap, name):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"{name} camera read failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t0 = grab_rgb(cap_top, "TOP")
    w0 = grab_rgb(cap_wrist, "WRIST")
    print(f"TOP   cam: {t0.shape} mean={t0.mean():.1f} std={t0.std():.1f}")
    print(f"WRIST cam: {w0.shape} mean={w0.mean():.1f} std={w0.std():.1f}   (std<3 => near-blank/dead feed)")

    # --- policy server ---
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    print(f"Connected to policy server {args.server_host}:{args.server_port}")

    # --- optional reset to a known start pose so the closed loop begins ON the demo manifold ---
    if args.home.strip().lower() != "none":
        home = np.array([float(x) for x in args.home.split(",")], dtype=np.float32)
        cur = np.array([float(robot.get_observation()[f"{j}.pos"]) for j in JOINTS], dtype=np.float32)
        nramp = max(1, int(args.reset_secs * args.hz))
        print(f"resetting to home {home.tolist()} over {args.reset_secs}s ...")
        for k in range(1, nramp + 1):
            q = cur + (home - cur) * (k / nramp)
            robot.send_action({f"{j}.pos": float(q[i]) for i, j in enumerate(JOINTS)})
            time.sleep(1.0 / args.hz)
        time.sleep(0.5)
        print("at home; starting policy.")

    period = 1.0 / args.hz
    q_prev = None  # last sent target, for optional EMA smoothing of the arm joints
    try:
        for cyc in range(args.max_cycles):
            obs = robot.get_observation()
            state = np.array([float(obs[f"{j}.pos"]) for j in JOINTS], dtype=np.float32)
            image = grab_rgb(cap_top, "TOP")
            wrist = grab_rgb(cap_wrist, "WRIST")
            result = policy.infer({
                "observation/image": image,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": args.prompt,
            })
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None]
            n = min(args.exec_horizon, len(actions))
            if cyc % 2 == 0:
                print(f"cyc {cyc:3d} top(m={image.mean():5.1f},s={image.std():4.1f}) "
                      f"wr(m={wrist.mean():5.1f},s={wrist.std():4.1f}) "
                      f"state=[{','.join(f'{x:+.1f}' for x in state)}]  "
                      f"act0=[{','.join(f'{x:+.1f}' for x in actions[0])}]")
            for a in actions[:n]:
                if args.smooth < 1.0 and q_prev is not None:
                    q = np.asarray(a, dtype=np.float32).copy()
                    q[:5] = args.smooth * q[:5] + (1.0 - args.smooth) * q_prev[:5]  # EMA arm joints; gripper crisp
                else:
                    q = np.asarray(a, dtype=np.float32)
                robot.send_action({f"{j}.pos": float(q[i]) for i, j in enumerate(JOINTS)})
                q_prev = q
                time.sleep(period)
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        cap_top.release()
        cap_wrist.release()
        robot.disconnect()
        print("done.")


if __name__ == "__main__":
    main()
