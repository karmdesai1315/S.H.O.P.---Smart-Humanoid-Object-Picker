"""Unified environment configuration for goal-conditioned G1 walking.

Location:
    source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/
    g1/29dof/aisle_env_cfg.py

Stage 1:
    - Train from scratch on an empty floor.
    - Sample target XYZ positions.
    - Walk toward a floor-level goal.
    - Stop within a 0.5 m radius.
    - No keyboard velocity commands.
    - No target heading.
    - No cameras or aisle assets spawned.

Stage 2:
    - Reuse this file and the same observation/action structure.
    - Enable the aisle assets.
    - Add camera-based waypoint planning and apple-target integration.

The aisle scene is defined here but disabled by default during Stage 1.
Changing enable_aisle alone does not implement Stage 2 navigation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    ObservationsCfg as WalkingObservationsCfg,
    RewardsCfg as WalkingRewardsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
    TerminationsCfg as WalkingTerminationsCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Aisle dimensions and appearance.
# Robot starts facing +X; left shelf is on +Y.
# ---------------------------------------------------------------------------

AISLE_HALF_WIDTH = 0.70
SHELF_DEPTH = 0.40
SHELF_LENGTH = 2.00
SHELF_CENTER_X = 2.00
BOARD_THICKNESS = 0.03
SHELF_HEIGHTS = (0.40, 0.80, 1.10)
UNIT_HEIGHT = 1.40

SHELF_COLOR = (0.75, 0.75, 0.78)
PANEL_COLOR = (0.55, 0.55, 0.60)
TARGET_COLOR = (0.85, 0.20, 0.15)


# ---------------------------------------------------------------------------
# Camera settings.
# ---------------------------------------------------------------------------

CAMERA_PITCH_DEG = 20.0
_pitch = math.radians(CAMERA_PITCH_DEG)

CAMERA_ROT = (
    math.cos(_pitch / 2),
    0.0,
    math.sin(_pitch / 2),
    0.0,
)


# ---------------------------------------------------------------------------
# Shared goal/stopping settings.
# ---------------------------------------------------------------------------

GOAL_RADIUS = 0.5
HOLD_RELEASE_RADIUS = 0.6

LINEAR_STOP_THRESHOLD = 0.08  # m/s
ANGULAR_STOP_THRESHOLD = 0.10  # rad/s
SETTLE_TIME_S = 0.5


# ---------------------------------------------------------------------------
# Scene helpers.
# ---------------------------------------------------------------------------

def static_box(
    prim_name: str,
    size: tuple,
    pos: tuple,
    color: tuple,
) -> AssetBaseCfg:
    """Create a fixed collidable box."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Aisle_" + prim_name,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def board(side: int, level: int) -> AssetBaseCfg:
    """Create one horizontal shelf board."""
    y = side * (AISLE_HALF_WIDTH + SHELF_DEPTH / 2)
    name = f"{'left' if side > 0 else 'right'}_board_{level}"

    return static_box(
        name,
        size=(SHELF_LENGTH, SHELF_DEPTH, BOARD_THICKNESS),
        pos=(SHELF_CENTER_X, y, SHELF_HEIGHTS[level]),
        color=SHELF_COLOR,
    )


def back_panel(side: int) -> AssetBaseCfg:
    """Create a shelf back panel."""
    y = side * (AISLE_HALF_WIDTH + SHELF_DEPTH + 0.01)
    name = f"{'left' if side > 0 else 'right'}_back_panel"

    return static_box(
        name,
        size=(SHELF_LENGTH, 0.02, UNIT_HEIGHT),
        pos=(SHELF_CENTER_X, y, UNIT_HEIGHT / 2),
        color=PANEL_COLOR,
    )


def shelf_top(level: int) -> float:
    return SHELF_HEIGHTS[level] + BOARD_THICKNESS / 2


APPLE_RADIUS = 0.04
APPLE_LEVEL = 1

APPLE_POS = (
    SHELF_CENTER_X,
    AISLE_HALF_WIDTH + 0.10,
    shelf_top(APPLE_LEVEL) + APPLE_RADIUS + 0.005,
)

YCB_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"


def grocery_item(
    name: str,
    usd_file: str,
    side: int,
    level: int,
    x: float,
) -> RigidObjectCfg:
    """Create a grocery product slightly above its shelf."""
    y = side * (AISLE_HALF_WIDTH + 0.12)

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{YCB_DIR}/{usd_file}",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(x, y, shelf_top(level) + 0.12),
        ),
    )


# ---------------------------------------------------------------------------
# Full aisle scene.
# Optional assets are removed from the scene config for Stage 1.
# ---------------------------------------------------------------------------

@configclass
class AisleSceneCfg(RobotSceneCfg):
    """Robot scene with shelves, grocery objects, and a head RGB-D camera."""

    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.2,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0576, 0.0175, 0.4299),
            rot=CAMERA_ROT,
            convention="world",
        ),
    )

    left_board_0 = board(+1, 0)
    left_board_1 = board(+1, 1)
    left_board_2 = board(+1, 2)
    left_back_panel = back_panel(+1)

    right_board_0 = board(-1, 0)
    right_board_1 = board(-1, 1)
    right_board_2 = board(-1, 2)
    right_back_panel = back_panel(-1)

    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetApple",
        spawn=sim_utils.SphereCfg(
            radius=APPLE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=TARGET_COLOR,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=APPLE_POS,
        ),
    )

    cracker_box = grocery_item(
        "CrackerBox",
        "003_cracker_box.usd",
        side=+1,
        level=1,
        x=1.5,
    )

    sugar_box = grocery_item(
        "SugarBox",
        "004_sugar_box.usd",
        side=+1,
        level=0,
        x=2.3,
    )

    soup_can = grocery_item(
        "SoupCan",
        "005_tomato_soup_can.usd",
        side=-1,
        level=1,
        x=1.8,
    )

    mustard = grocery_item(
        "Mustard",
        "006_mustard_bottle.usd",
        side=-1,
        level=2,
        x=2.4,
    )


# ---------------------------------------------------------------------------
# Goal-aware walking reward helpers.
# ---------------------------------------------------------------------------

def goal_gait_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Encourage stepping during travel and both feet down during hold."""
    term = env.command_manager.get_term(command_name)

    # Disable the original velocity-command gate.
    walking_reward = mdp.feet_gait(
        env,
        period=0.8,
        offset=[0.0, 0.5],
        threshold=0.55,
        sensor_cfg=sensor_cfg,
        command_name=None,
    )

    sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    )

    # Match the maximum two-foot gait reward scale.
    standing_reward = contacts.all(dim=-1).float() * 2.0

    # Include live distance because rewards precede command refresh.
    should_hold = (
        term.goal_distance_xy <= term.cfg.goal_radius
    ) | term.hold_position

    return torch.where(
        should_hold,
        standing_reward,
        walking_reward,
    )


def successful_goal_event(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Completion event, excluding simultaneous fall terminations."""
    manager = env.termination_manager

    success = manager.get_term("goal_reached")
    fallen = (
        manager.get_term("base_height")
        | manager.get_term("bad_orientation")
    )

    # RewardManager multiplies by step_dt. Dividing here makes the
    # configured reward weight an actual once-per-episode event bonus.
    return (success & ~fallen).float() / env.step_dt


def falling_event(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Fall event, excluding timeouts and ordinary goal completion."""
    manager = env.termination_manager

    fallen = (
        manager.get_term("base_height")
        | manager.get_term("bad_orientation")
    )

    return fallen.float() / env.step_dt


# ---------------------------------------------------------------------------
# Goal commands.
# ---------------------------------------------------------------------------

@configclass
class GoalCommandsCfg:
    goal_position = mdp.GoalPositionCommandCfg(
        asset_name="robot",
        distance_range=(0.75, 1.25),
        target_height_range=(0.6, 1.2),
        floor_height=0.0,
        goal_radius=GOAL_RADIUS,
        hold_release_radius=HOLD_RELEASE_RADIUS,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
    )


# ---------------------------------------------------------------------------
# Observations.
#
# Inherit robot observations from the original walking configuration.
# Remove velocity commands and add goal/waypoint observations.
# Keep this layout unchanged when proceeding to Stage 2.
# ---------------------------------------------------------------------------

@configclass
class GoalPolicyObservationsCfg(WalkingObservationsCfg.PolicyCfg):
    velocity_commands = None

    base_velocity_xy = ObsTerm(
        func=mdp.base_linear_velocity_xy,
        scale=1.0,
    )

    waypoint_xy = ObsTerm(
        func=mdp.local_waypoint_xy,
        params={"command_name": "goal_position"},
        scale=0.5,
    )

    target_xyz = ObsTerm(
        func=mdp.target_position_xyz,
        params={"command_name": "goal_position"},
        scale=0.5,
    )

    hold_position = ObsTerm(
        func=mdp.hold_position_observation,
        params={"command_name": "goal_position"},
    )

    def __post_init__(self):
        super().__post_init__()
        self.history_length = 5
        self.flatten_history_dim = True
        self.concatenate_terms = True
        self.enable_corruption = True


@configclass
class GoalCriticObservationsCfg(GoalPolicyObservationsCfg):
    """Same observation layout as the actor, without observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class GoalObservationsCfg:
    policy: GoalPolicyObservationsCfg = GoalPolicyObservationsCfg()
    critic: GoalCriticObservationsCfg = GoalCriticObservationsCfg()


# ---------------------------------------------------------------------------
# Rewards.
# ---------------------------------------------------------------------------

@configclass
class GoalRewardsCfg(WalkingRewardsCfg):
    # Remove rewards requiring a velocity-command generator.
    track_lin_vel_xy = None
    track_ang_vel_z = None

    goal_distance_and_stop = RewTerm(
        func=mdp.goal_distance_and_stop_reward,
        weight=2.0,
        params={
            "command_name": "goal_position",
            "linear_speed_threshold": LINEAR_STOP_THRESHOLD,
            "angular_speed_threshold": ANGULAR_STOP_THRESHOLD,
        },
    )

    motion_at_goal = RewTerm(
        func=mdp.motion_inside_goal_penalty,
        weight=-2.0,
        params={
            "command_name": "goal_position",
            "angular_weight": 0.25,
        },
    )

    goal_completion = RewTerm(
        func=successful_goal_event,
        weight=60.0,
    )

    fall = RewTerm(
        func=falling_event,
        weight=-5.0,
    )

    # Replace the original base_velocity-dependent gait reward.
    gait = RewTerm(
        func=goal_gait_reward,
        weight=0.2,
        params={
            "command_name": "goal_position",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*ankle_roll.*",
            ),
        },
    )


# ---------------------------------------------------------------------------
# Terminations.
# ---------------------------------------------------------------------------

@configclass
class GoalTerminationsCfg(WalkingTerminationsCfg):
    # Inherited fall checks:
    #   base_height
    #   bad_orientation

    # Replace the inherited timeout.
    time_out = DoneTerm(
        func=mdp.goal_timeout,
        time_out=True,
        params={"timeout_s": 20.0},
    )

    goal_reached = DoneTerm(
        func=mdp.GoalReached,
        time_out=False,
        params={
            "command_name": "goal_position",
            "linear_speed_threshold": LINEAR_STOP_THRESHOLD,
            "angular_speed_threshold": ANGULAR_STOP_THRESHOLD,
            "settle_time_s": SETTLE_TIME_S,
        },
    )


# ---------------------------------------------------------------------------
# Curriculum.
# ---------------------------------------------------------------------------

@configclass
class GoalCurriculumCfg:
    """No terrain or velocity-command curriculum.

    Start with nearby goals. A success-based goal-distance curriculum
    can be added after basic walking and stopping have been verified.
    """

    pass


# ---------------------------------------------------------------------------
# Unified training environment.
# ---------------------------------------------------------------------------

@configclass
class AisleEnvCfg(RobotEnvCfg):
    """Goal-conditioned walking environment.

    Stage 1 uses sampled numerical targets on an empty floor.
    The shelf apple is not the Stage 1 target.

    Aisle assets remain defined in this file for Stage 2.
    """

    enable_aisle: bool = False

    scene: AisleSceneCfg = AisleSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
    )

    commands: GoalCommandsCfg = GoalCommandsCfg()
    observations: GoalObservationsCfg = GoalObservationsCfg()
    rewards: GoalRewardsCfg = GoalRewardsCfg()
    terminations: GoalTerminationsCfg = GoalTerminationsCfg()
    curriculum: GoalCurriculumCfg = GoalCurriculumCfg()

    def __post_init__(self):
        # Scene configs are edited before the actual scene is constructed.
        # Stage 1 does not spawn shelves, products, apple, or camera.
        if not self.enable_aisle:
            for name in (
                "left_board_0",
                "left_board_1",
                "left_board_2",
                "left_back_panel",
                "right_board_0",
                "right_board_1",
                "right_board_2",
                "right_back_panel",
                "target_object",
                "cracker_box",
                "sugar_box",
                "soup_can",
                "mustard",
                "head_camera",
            ):
                setattr(self.scene, name, None)

        super().__post_init__()

        # Flat floor.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None

        # Keep episode and goal timeout settings consistent.
        self.episode_length_s = 20.0
        self.terminations.time_out.params["timeout_s"] = (
            self.episode_length_s
        )

        # Simplify learning from scratch.
        self.events.push_robot = None
        self.events.add_base_mass = None

        self.events.physics_material.params["static_friction_range"] = (
            0.6, 1.0
        )
        self.events.physics_material.params["dynamic_friction_range"] = (
            0.6, 1.0
        )

        # Fixed XY start with varied yaw.
        # No desired heading is included in the observations.
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }

        self.events.reset_robot_joints.params["velocity_range"] = (
            -0.1, 0.1
        )

        # Starting weights for inherited walking rewards.
        self.rewards.alive.weight = 0.05

        # Body stability and limits.
        self.rewards.base_linear_velocity.weight = -2.0
        self.rewards.base_angular_velocity.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height.weight = -10.0
        self.rewards.dof_pos_limits.weight = -5.0

        # Motion regularization.
        self.rewards.joint_vel.weight = -0.0002
        self.rewards.joint_acc.weight = -2.5e-7
        self.rewards.action_rate.weight = -0.02
        self.rewards.energy.weight = -2e-5

        # Posture.
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_waists.weight = -0.5
        self.rewards.joint_deviation_legs.weight = -0.3

        # Feet and non-foot contacts.
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_clearance.weight = 0.2
        self.rewards.undesired_contacts.weight = -1.0


# ---------------------------------------------------------------------------
# Playback configuration, in the same file.
# ---------------------------------------------------------------------------

@configclass
class AislePlayEnvCfg(AisleEnvCfg):
    """Single-environment playback of the goal-conditioned policy."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)