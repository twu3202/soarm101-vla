"""Validate the served SO-101 policy end-to-end (no arm): connect, send a dummy obs, check
the returned action chunk has shape (action_horizon, 6). Run with `uv run python` in openpi env."""
import numpy as np
from openpi_client import websocket_client_policy

c = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
print("server metadata:", c.get_server_metadata())

obs = {
    "observation/image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    "observation/state": (np.random.rand(6).astype(np.float32) * 20 - 10),  # ~joint deg
    "prompt": "Sort the cubes into a row left to right: red, orange, yellow, green, blue.",
}
r = c.infer(obs)
a = np.asarray(r["actions"])
print("RETURNED actions shape:", a.shape, "dtype:", a.dtype)
print("action[0] (6-dim, abs joint deg):", np.round(a[0], 2).tolist())
assert a.ndim == 2 and a.shape[1] == 6, f"expected (H,6), got {a.shape}"
print("INFER OK")
