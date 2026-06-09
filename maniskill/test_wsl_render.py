"""Test SAPIEN render in WSL. Run inside WSL sapien env."""
import sys
import threading

sys.path.insert(0, '/mnt/d/soarm101_sorting/maniskill')

print('1. Importing libraries...', flush=True)
import sapien
print(f'   sapien {sapien.__version__}', flush=True)
import mani_skill
print(f'   mani_skill {mani_skill.__version__}', flush=True)
import mani_skill.envs
import sort_cubes_env  # noqa
import gymnasium as gym
import torch

print('2. Creating env (num_envs=1)...', flush=True)
env = gym.make(
    'SortCubesSO100-v1',
    num_envs=1,
    obs_mode='state',
    render_mode='rgb_array',
    sim_backend='physx_cuda',
    num_active_cubes=5,
)
print('3. Reset...', flush=True)
obs, _ = env.reset(seed=0)
print(f'   obs shape: {obs.shape if hasattr(obs, "shape") else type(obs)}', flush=True)

print('4. Attempting render (15s timeout)...', flush=True)
result = [None]; err = [None]
def do_render():
    try:
        result[0] = env.render()
    except Exception as e:
        err[0] = e
t = threading.Thread(target=do_render, daemon=True)
t.start()
t.join(timeout=15)
if t.is_alive():
    print('   *** RENDER HUNG ***', flush=True)
elif err[0]:
    print(f'   *** ERROR: {type(err[0]).__name__}: {err[0]} ***', flush=True)
else:
    img = result[0]
    if hasattr(img, 'cpu'):
        import numpy as np
        arr = img.cpu().numpy()
    else:
        import numpy as np
        arr = np.asarray(img)
    print(f'   *** RENDER OK *** shape={arr.shape} dtype={arr.dtype}', flush=True)
    # Save PNG
    try:
        from PIL import Image
        img2d = arr.squeeze() if arr.ndim == 4 else arr
        Image.fromarray(img2d.astype('uint8')).save('/mnt/d/soarm101_sorting/maniskill/test_render_wsl.png')
        print('   Saved /mnt/d/soarm101_sorting/maniskill/test_render_wsl.png', flush=True)
    except Exception as e:
        print(f'   PIL save error: {e}', flush=True)
env.close()
print('Done.', flush=True)
