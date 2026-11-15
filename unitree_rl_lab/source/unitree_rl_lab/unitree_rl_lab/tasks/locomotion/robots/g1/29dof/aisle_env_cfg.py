"""S.H.O.P. environment config: scene, numbers and stage classes, all in one file.

File: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/aisle_env_cfg.py

Nothing here comes from velocity_env_cfg.py. Every function comes from one of our files in mdp/:
  goal_commands.py      cmd.    goal_position, base_velocity (guide), obstacles
  goal_observations.py  obs.    what the policy and critic see
  goal_rewards.py       rew.    Unitree walking rewards + goal rewards
  goal_terminations.py  done.   time-out and falls
Events (resets, randomization) and the joint action come from Isaac Lab itself (isaac_mdp.).

Stage 1: flat floor, random goal 0.5-1.5 m away in any direction. Walk there and stand inside the circle.
  Stage1EnvCfg        training (4096 robots)
  Stage1PlayEnvCfg    watch it (16 robots, goal markers on)

The aisle scene (shelves, products, YOLO items, head camera) is in AisleSceneCfg, for later stages.

Robot frame reminder: x forward, y left, z up. The robot starts at its env origin with a random heading.
"""

import math

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion.mdp import goal_commands as cmd
from unitree_rl_lab.tasks.locomotion.mdp import goal_observations as obs
from unitree_rl_lab.tasks.locomotion.mdp import goal_rewards as rew
from unitree_rl_lab.tasks.locomotion.mdp import goal_terminations as done


# ===========================================================================
# Numbers (change things here first)
# ===========================================================================
EPISODE_LENGTH_S = 20.0

# goal
GOAL_DISTANCE_RANGE = (0.5, 1.5)  # m from the start, any direction (your curriculum can change this)
GOAL_RADIUS = 0.4  # inside = at the goal
HOLD_RELEASE_RADIUS = 0.5  # stop flag stays on until the robot drifts past this
TARGET_HEIGHT_RANGE = (0.6, 1.2)  # gripper target height above the floor (m)

# guide velocity (what the walking rewards track)
GUIDE_MIN_SPEED = 0.2  # m/s, raise to 0.3 if it stalls just outside the circle
GUIDE_MAX_SPEED = 0.6
GUIDE_MAX_BACKWARD = 0.3
GUIDE_MAX_SIDE = 0.3
GUIDE_MAX_TURN = 0.5  # rad/s
GUIDE_TURN_MIN_DIST = 0.6  # closer than this: step there without turning

# obstacles
NUM_OBSTACLE_SLOTS = 4

# observation history (frames) for the robot part; task part is 1 frame
PROPRIO_HISTORY = 5

# "standing still" for at_goal_still
STILL_LIN_SPEED = 0.1  # m/s
STILL_YAW_SPEED = 0.15  # rad/s

# body names used by several rewards
FEET = ".*ankle_roll.*"


# ===========================================================================
# Scene
# ===========================================================================
@configclass
class ShopSceneCfg(InteractiveSceneCfg):
    """Flat floor, the G1, contact sensors and a sky light (Stage 1)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # plain G1, 29 body joints (swap in G1_INSPIRE_CFG later for the hands)
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---- aisle (later stages) -------------------------------------------------
# Layout from above, robot starts at the origin facing +x:
#        y ^   [==== left shelf unit ====]   y = +0.70 .. +1.10
#  robot o-->   aisle (1.4 m wide)            x = 1.0 .. 3.0
#          |   [==== right shelf unit ===]   y = -0.70 .. -1.10
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

CAMERA_PITCH_DEG = 20.0  # head camera tilt down (real G1 D435 is 47.6)
_pitch = math.radians(CAMERA_PITCH_DEG)
CAMERA_ROT = (math.cos(_pitch / 2), 0.0, math.sin(_pitch / 2), 0.0)

YCB_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"  # products with physics
YCB_VISUAL_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned"  # fixed props YOLO can recognise
PROP_ORIGIN_AT_CENTER = True  # if props float or sink into the shelf, flip this


def shelf_top(level: int) -> float:
    return SHELF_HEIGHTS[level] + BOARD_THICKNESS / 2


APPLE_RADIUS = 0.04
APPLE_POS = (SHELF_CENTER_X, AISLE_HALF_WIDTH + 0.10, shelf_top(1) + APPLE_RADIUS + 0.005)


def static_box(prim_name: str, size: tuple, pos: tuple, color: tuple) -> AssetBaseCfg:
    """A fixed box the robot can bump into."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Aisle_" + prim_name,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def board(side: int, level: int) -> AssetBaseCfg:
    """One shelf board. side = +1 left, -1 right."""
    y = side * (AISLE_HALF_WIDTH + SHELF_DEPTH / 2)
    name = f"{'left' if side > 0 else 'right'}_board_{level}"
    return static_box(name, (SHELF_LENGTH, SHELF_DEPTH, BOARD_THICKNESS), (SHELF_CENTER_X, y, SHELF_HEIGHTS[level]), SHELF_COLOR)


def back_panel(side: int) -> AssetBaseCfg:
    y = side * (AISLE_HALF_WIDTH + SHELF_DEPTH + 0.01)
    name = f"{'left' if side > 0 else 'right'}_back_panel"
    return static_box(name, (SHELF_LENGTH, 0.02, UNIT_HEIGHT), (SHELF_CENTER_X, y, UNIT_HEIGHT / 2), PANEL_COLOR)


def grocery_item(name: str, usd_file: str, side: int, level: int, x: float) -> RigidObjectCfg:
    """A YCB product with physics, dropped just above the shelf."""
    y = side * (AISLE_HALF_WIDTH + 0.12)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.UsdFileCfg(usd_path=f"{YCB_DIR}/{usd_file}"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, shelf_top(level) + 0.12)),
    )


def shelf_prop(name: str, usd_file: str, side: int, level: int, x: float, height: float) -> AssetBaseCfg:
    """A fixed (no physics) YCB item standing on a shelf. height = item height (m)."""
    y = side * (AISLE_HALF_WIDTH + 0.12)
    z = shelf_top(level) + (height / 2 if PROP_ORIGIN_AT_CENTER else 0.0) + 0.002
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.UsdFileCfg(usd_path=f"{YCB_VISUAL_DIR}/{usd_file}"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, z)),
    )


@configclass
class AisleSceneCfg(ShopSceneCfg):
    """Stage 1 scene + shelves, products, YOLO items and the head camera (run with --enable_cameras)."""

    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=15.2, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0576, 0.0175, 0.4299), rot=CAMERA_ROT, convention="world"),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TARGET_COLOR),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=APPLE_POS),
    )

    cracker_box = grocery_item("CrackerBox", "003_cracker_box.usd", side=+1, level=1, x=1.5)
    sugar_box = grocery_item("SugarBox", "004_sugar_box.usd", side=+1, level=0, x=2.3)
    soup_can = grocery_item("SoupCan", "005_tomato_soup_can.usd", side=-1, level=1, x=1.8)
    mustard = grocery_item("Mustard", "006_mustard_bottle.usd", side=-1, level=2, x=2.4)

    banana = shelf_prop("Banana", "011_banana.usd", side=+1, level=1, x=2.5, height=0.04)
    mug_left = shelf_prop("MugLeft", "025_mug.usd", side=+1, level=0, x=1.5, height=0.08)
    bowl = shelf_prop("Bowl", "024_bowl.usd", side=+1, level=2, x=2.0, height=0.055)
    bleach = shelf_prop("Bleach", "021_bleach_cleanser.usd", side=-1, level=1, x=2.4, height=0.25)
    mug_right = shelf_prop("MugRight", "025_mug.usd", side=-1, level=1, x=1.3, height=0.08)


# ===========================================================================
# Commands (order matters: base_velocity reads goal_position)
# ===========================================================================
@configclass
class CommandsCfg:
    goal_position = cmd.GoalPositionCommandCfg(
        distance_range=GOAL_DISTANCE_RANGE,
        target_height_range=TARGET_HEIGHT_RANGE,
        goal_radius=GOAL_RADIUS,
        hold_release_radius=HOLD_RELEASE_RADIUS,
        debug_vis=False,
    )
    base_velocity = cmd.GuideVelocityCommandCfg(
        goal_command_name="goal_position",
        min_speed=GUIDE_MIN_SPEED,
        max_speed=GUIDE_MAX_SPEED,
        max_forward=GUIDE_MAX_SPEED,
        max_backward=GUIDE_MAX_BACKWARD,
        max_side=GUIDE_MAX_SIDE,
        max_turn=GUIDE_MAX_TURN,
        turn_min_dist=GUIDE_TURN_MIN_DIST,
    )
    obstacles = cmd.ObstacleCommandCfg(num_slots=NUM_OBSTACLE_SLOTS)


# ===========================================================================
# Actions: all 29 joints, target = default + 0.25 * action (same as Unitree)
# ===========================================================================
@configclass
class ActionsCfg:
    JointPositionAction = isaac_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# ===========================================================================
# Observations (fixed for all stages, see goal_observations.py)
# ===========================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # robot: PROPRIO_HISTORY frames each
        base_ang_vel = ObsTerm(func=obs.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=PROPRIO_HISTORY)
        projected_gravity = ObsTerm(func=obs.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=PROPRIO_HISTORY)
        joint_pos_rel = ObsTerm(func=obs.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=PROPRIO_HISTORY)
        joint_vel_rel = ObsTerm(func=obs.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=PROPRIO_HISTORY)
        last_action = ObsTerm(func=obs.last_action, history_length=PROPRIO_HISTORY)
        # task: 1 frame
        goal_xy = ObsTerm(func=obs.goal_xy, params={"command_name": "goal_position"})
        waypoint_xy = ObsTerm(func=obs.waypoint_xy, params={"command_name": "goal_position"})
        target_pos = ObsTerm(func=obs.target_pos, params={"command_name": "goal_position"})
        target_quat = ObsTerm(func=obs.target_quat, params={"command_name": "goal_position"})
        hold_flag = ObsTerm(func=obs.hold_flag, params={"command_name": "goal_position"})
        obstacle_slots = ObsTerm(func=obs.obstacle_slots, params={"command_name": "obstacles"})

        def __post_init__(self):
            self.history_length = None  # None = each term uses its own history_length
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # same as the policy, no noise ...
        base_ang_vel = ObsTerm(func=obs.base_ang_vel, scale=0.2, history_length=PROPRIO_HISTORY)
        projected_gravity = ObsTerm(func=obs.projected_gravity, history_length=PROPRIO_HISTORY)
        joint_pos_rel = ObsTerm(func=obs.joint_pos_rel, history_length=PROPRIO_HISTORY)
        joint_vel_rel = ObsTerm(func=obs.joint_vel_rel, scale=0.05, history_length=PROPRIO_HISTORY)
        last_action = ObsTerm(func=obs.last_action, history_length=PROPRIO_HISTORY)
        goal_xy = ObsTerm(func=obs.goal_xy, params={"command_name": "goal_position"})
        waypoint_xy = ObsTerm(func=obs.waypoint_xy, params={"command_name": "goal_position"})
        target_pos = ObsTerm(func=obs.target_pos, params={"command_name": "goal_position"})
        target_quat = ObsTerm(func=obs.target_quat, params={"command_name": "goal_position"})
        hold_flag = ObsTerm(func=obs.hold_flag, params={"command_name": "goal_position"})
        obstacle_slots = ObsTerm(func=obs.obstacle_slots, params={"command_name": "obstacles"})
        # ... plus simulation-only extras
        base_lin_vel = ObsTerm(func=obs.base_lin_vel, history_length=PROPRIO_HISTORY)
        guide_velocity = ObsTerm(func=obs.guide_velocity, params={"command_name": "base_velocity"})
        goal_distance = ObsTerm(func=obs.goal_distance, params={"command_name": "goal_position"})

        def __post_init__(self):
            self.history_length = None
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ===========================================================================
# Rewards (weights are multiplied by step_dt = 0.02 inside Isaac Lab)
# ===========================================================================
@configclass
class RewardsCfg:
    # -- walking: track the guide velocity (Unitree weights)
    track_lin_vel_xy = RewTerm(
        func=rew.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=rew.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    alive = RewTerm(func=rew.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=rew.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=rew.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=rew.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=rew.base_height_l2, weight=-10.0, params={"target_height": 0.78})

    # -- joints
    joint_vel = RewTerm(func=rew.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=rew.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=rew.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=rew.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=rew.energy, weight=-2e-5)
    joint_deviation_arms = RewTerm(
        func=rew.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"])},
    )
    joint_deviation_waists = RewTerm(
        func=rew.joint_deviation_l1, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])}
    )
    joint_deviation_legs = RewTerm(
        func=rew.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- feet
    gait = RewTerm(
        func=rew.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
        },
    )
    feet_slide = RewTerm(
        func=rew.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
        },
    )
    feet_clearance = RewTerm(
        func=rew.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "command_name": "base_velocity",  # only pays while the guide says walk
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    undesired_contacts = RewTerm(
        func=rew.undesired_contacts,
        weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"])},
    )

    # -- goal (on top of walking)
    goal_progress = RewTerm(
        func=rew.GoalDistanceProgress, weight=4.0, params={"command_name": "goal_position", "flat_dist": 0.3}
    )
    fall = RewTerm(func=rew.fall_penalty, weight=-100.0, params={"term_names": ["base_height", "bad_orientation"]})
    at_goal_still = RewTerm(
        func=rew.at_goal_still,
        weight=1.0,
        params={"command_name": "goal_position", "lin_speed_max": STILL_LIN_SPEED, "yaw_speed_max": STILL_YAW_SPEED},
    )


# ===========================================================================
# Terminations (reaching the goal does NOT end the episode in Stage 1)
# ===========================================================================
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=done.time_out, time_out=True)
    base_height = DoneTerm(func=done.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=done.bad_orientation, params={"limit_angle": 0.8})


# ===========================================================================
# Events
# ===========================================================================
@configclass
class EventsCfg:
    # startup: floor friction per body
    physics_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset: at the env origin, random heading, default joint pose with a little joint speed
    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-0.1, 0.1)},
    )
    # no pushes and no extra torso mass in Stage 1 (add them back once it walks to goals)


# ===========================================================================
# Curriculum (you're writing this one; register terms here)
# ===========================================================================
@configclass
class CurriculumCfg:
    pass


# ===========================================================================
# Stage 1
# ===========================================================================
@configclass
class Stage1EnvCfg(ManagerBasedRLEnvCfg):
    """Training: walk to a random goal 0.5-1.5 m away on a flat floor, then stand in the circle."""

    scene: ShopSceneCfg = ShopSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # timing: physics 200 Hz, policy 50 Hz (same as Unitree)
        self.decimation = 4
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Stage1PlayEnvCfg(Stage1EnvCfg):
    """Watch Stage 1: a few robots, goal markers on (green ball = goal, small frame = gripper target)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
        self.commands.goal_position.debug_vis = True
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)


# old names, so the current task registration (Unitree-G1-29dof-Aisle) keeps working
AisleEnvCfg = Stage1EnvCfg
AislePlayEnvCfg = Stage1PlayEnvCfg