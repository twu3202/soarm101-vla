"""
================================================================================
ADD THESE TO openpi/src/openpi/training/config.py   (mirrors LeRobotLiberoDataConfig
+ pi05_eeg_libero_v4_lora, minus the EEG bits). NOT imported anywhere — reference only.
================================================================================

# --- (a) top of file, near the other policy imports ---
from openpi.policies import so101_policy

# --- (b) a new data-config factory, place right after `class LeRobotLiberoDataConfig` ---
"""

import dataclasses
import pathlib

from typing_extensions import override

from openpi.models import model as _model
from openpi.policies import so101_policy
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class LeRobotSo101DataConfig(DataConfigFactory):
    """SO-ARM101: single arm, single top-down camera, 6-DOF (5 joints + gripper).
    Standard LeRobot keys (observation.images.top / observation.state / action)."""

    # Convert absolute joint targets -> deltas vs current state (gripper stays absolute),
    # matching what the pi base models expect (same as LeRobotAlohaDataConfig).
    use_delta_joint_actions: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
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
            # 6 dims = [pan, lift, elbow, wrist_flex, wrist_roll, gripper];
            # first 5 -> delta, gripper -> absolute.
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


# --- (c) a new TrainConfig entry, add inside the `_CONFIGS = [ ... ]` list ---
#
#     TrainConfig(
#         name="pi05_so101_sort_lora",
#         model=pi0_config.Pi0Config(
#             pi05=True,
#             action_horizon=10,
#             discrete_state_input=False,
#             paligemma_variant="gemma_2b_lora",
#             action_expert_variant="gemma_300m_lora",
#         ),
#         data=LeRobotSo101DataConfig(
#             repo_id="local/so101_sort_cubes",
#             base_config=DataConfig(prompt_from_task=True),
#         ),
#         batch_size=32,
#         weight_loader=weight_loaders.CheckpointWeightLoader(
#             "gs://openpi-assets/checkpoints/pi05_base/params"
#         ),
#         num_train_steps=10_000,
#         freeze_filter=pi0_config.Pi0Config(
#             pi05=True, action_horizon=10, discrete_state_input=False,
#             paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
#         ).get_freeze_filter(),
#         ema_decay=None,
#     ),
