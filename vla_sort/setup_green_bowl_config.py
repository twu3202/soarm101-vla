"""Idempotently patch openpi/src/openpi/training/config.py to register the 2-CAMERA
green-cube-into-bowl pi0.5 LoRA config. Pure file I/O (run with system python).
Backs up to config.py.bak_greenbowl. NO EEG.

Adds alongside (does NOT replace) the existing 1-cam pi05_so101_sort_lora:
  * LeRobotGreenBowlDataConfig  — repacks BOTH cameras (top + wrist)
  * TrainConfig pi05_green_bowl_lora — repo_id local/so101_green_bowl

Run from the openpi repo root:  python ~/Projects/so101_sort/setup_green_bowl_config.py
"""
import pathlib
import re
import shutil
import sys

CFG = pathlib.Path("src/openpi/training/config.py")
src = CFG.read_text()

if "pi05_green_bowl_lora" in src:
    print("[setup] already patched — nothing to do")
    sys.exit(0)

shutil.copy(CFG, str(CFG) + ".bak_greenbowl")
print(f"[setup] backed up -> {CFG}.bak_greenbowl")

# --- 1) ensure the so101 policy module is imported (setup_so101_config may have already done it) ---
if "import openpi.policies.so101_policy as so101_policy" not in src:
    imp_anchor = "import openpi.policies.libero_policy as libero_policy\n"
    if imp_anchor not in src:
        print("[setup] FATAL: libero_policy import anchor not found"); sys.exit(2)
    src = src.replace(imp_anchor, imp_anchor + "import openpi.policies.so101_policy as so101_policy\n", 1)
    print("[setup] added so101_policy import")

# --- 2) data config class + TrainConfig appended right before the _CONFIGS consumer ---
BLOCK = '''
# ============== green-cube-into-bowl, 2 cameras (added by setup_green_bowl_config.py) ==============
@dataclasses.dataclass(frozen=True)
class LeRobotGreenBowlDataConfig(DataConfigFactory):
    """SO-ARM101 single arm, 6-DOF, TWO cameras: top (overhead) + wrist (gripper).
    LeRobot keys observation.images.top / observation.images.wrist / observation.state / action."""

    use_delta_joint_actions: bool = True

    @override
    def create(self, assets_dirs, model_config):
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.top",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[so101_policy.So101Inputs(model_type=model_config.model_type)],
            outputs=[so101_policy.So101Outputs()],
        )
        if self.use_delta_joint_actions:
            # 6 dims = [pan, lift, elbow, wrist_flex, wrist_roll, gripper]: first 5 -> delta, gripper absolute.
            delta_action_mask = _transforms.make_bool_mask(5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


_CONFIGS.append(
    TrainConfig(
        name="pi05_green_bowl_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotGreenBowlDataConfig(
            repo_id="local/so101_green_bowl",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=8_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    )
)
# =============================================================================================

'''

anchors = [
    r"^_CONFIGS_DICT\b",
    r"^\s*_CONFIGS_DICT\b",
    r"for config in _CONFIGS",
    r"for c in _CONFIGS",
    r"^def _get_config\b",
    r"^def get_config\b",
]
pos = None
for a in anchors:
    m = re.search(a, src, re.M)
    if m:
        pos = src.rfind("\n", 0, m.start()) + 1
        print(f"[setup] insert anchor matched: {a!r} at offset {pos}")
        break

if pos is None:
    print("[setup] FATAL: could not find _CONFIGS consumer. Tail of file:")
    print("\n".join(src.splitlines()[-50:]))
    sys.exit(3)

src = src[:pos] + BLOCK + src[pos:]
CFG.write_text(src)
print("[setup] patched config.py OK (added LeRobotGreenBowlDataConfig + pi05_green_bowl_lora)")
