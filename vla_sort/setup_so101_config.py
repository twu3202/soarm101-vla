"""Idempotently patch openpi/src/openpi/training/config.py to register the SO-101 cube-sort
pi0.5 LoRA config. Pure file I/O (run with the system python). Backs up to config.py.bak_so101.
NO EEG: this only adds a clean single-arm LeRobot data config + one TrainConfig.

Run from the openpi repo root:  python ~/Projects/so101_sort/setup_so101_config.py
"""
import pathlib
import re
import shutil
import sys

CFG = pathlib.Path("src/openpi/training/config.py")
src = CFG.read_text()

if "pi05_so101_sort_lora" in src:
    print("[setup] already patched — nothing to do")
    sys.exit(0)

shutil.copy(CFG, str(CFG) + ".bak_so101")
print(f"[setup] backed up -> {CFG}.bak_so101")

# --- 1) import the so101 policy module (after the libero policy import) ---
imp_anchor = "import openpi.policies.libero_policy as libero_policy\n"
if imp_anchor not in src:
    print("[setup] FATAL: libero_policy import anchor not found"); sys.exit(2)
src = src.replace(imp_anchor, imp_anchor + "import openpi.policies.so101_policy as so101_policy\n", 1)

# --- 2) data config class + TrainConfig appended right before the _CONFIGS consumer ---
BLOCK = '''
# ===================== SO-ARM101 cube-sort (added by setup_so101_config.py) =====================
@dataclasses.dataclass(frozen=True)
class LeRobotSo101DataConfig(DataConfigFactory):
    """SO-ARM101: single arm, single top-down camera, 6-DOF (5 joints + gripper).
    Standard LeRobot keys (observation.images.top / observation.state / action)."""

    use_delta_joint_actions: bool = True

    @override
    def create(self, assets_dirs, model_config):
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.top",
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
        name="pi05_so101_sort_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotSo101DataConfig(
            repo_id="local/so101_sort_cubes",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10_000,
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

# Find where _CONFIGS is consumed (dict built) and insert our block just before it.
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
        # back up to the start of that line
        pos = src.rfind("\n", 0, m.start()) + 1
        print(f"[setup] insert anchor matched: {a!r} at offset {pos}")
        break

if pos is None:
    print("[setup] FATAL: could not find _CONFIGS consumer. Tail of file:")
    print("\n".join(src.splitlines()[-50:]))
    sys.exit(3)

src = src[:pos] + BLOCK + src[pos:]
CFG.write_text(src)
print("[setup] patched config.py OK (added LeRobotSo101DataConfig + pi05_so101_sort_lora)")
