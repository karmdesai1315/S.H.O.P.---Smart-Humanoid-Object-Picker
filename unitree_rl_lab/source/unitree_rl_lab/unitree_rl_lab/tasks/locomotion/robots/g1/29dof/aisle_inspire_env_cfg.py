"""Grocery aisle with a G1 that has Inspire 5-finger hands.

Place this file next to aisle_env_cfg.py:
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/aisle_inspire_env_cfg.py

Same aisle, camera and objects as aisle_env_cfg.py. Differences:
  - the robot model has Inspire hands (53 joints instead of 29)
  - the walking policy still only sees and controls the 29 body joints, so the policy
    trained on the plain G1 loads and runs unchanged
  - the fingers are controlled separately by setting their joint targets directly
    (see scripts/rsl_rl/play_hands.py)
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.g1_inspire import G1_BODY_JOINTS, G1_INSPIRE_CFG

from .aisle_env_cfg import AisleEnvCfg, AisleSceneCfg


@configclass
class AisleInspireSceneCfg(AisleSceneCfg):
    """The aisle scene, with the G1 + Inspire hands robot."""

    robot = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class AisleInspireEnvCfg(AisleEnvCfg):
    """Aisle task with Inspire hands. The policy only handles the 29 body joints."""

    scene: AisleInspireSceneCfg = AisleInspireSceneCfg(num_envs=1, env_spacing=5.0)

    def __post_init__(self):
        super().__post_init__()

        # Only the 29 body joints, so the policy's inputs and outputs keep the sizes it was trained with.
        # Each term gets its OWN selector: Isaac Lab fills one in when first used, so sharing breaks.
        self.actions.JointPositionAction.joint_names = G1_BODY_JOINTS
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos_rel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_JOINTS)}
            group.joint_vel_rel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=G1_BODY_JOINTS)}
