import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
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
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


AISLE_HALF_WIDTH = 0.70  # distance from aisle centre line to shelf front edge
SHELF_DEPTH = 0.40
SHELF_LENGTH = 2.00
SHELF_CENTER_X = 2.00  # shelves run from x = 1.0 to x = 3.0
BOARD_THICKNESS = 0.03
SHELF_HEIGHTS = (0.40, 0.80, 1.10)  # top surface of each board is height + thickness/2
UNIT_HEIGHT = 1.40  # height of the back panel

# Head camera downward tilt in degrees (0 = straight ahead). The real G1's D435 is 47.6.
CAMERA_PITCH_DEG = 20.0
_pitch = math.radians(CAMERA_PITCH_DEG)
CAMERA_ROT = (math.cos(_pitch / 2), 0.0, math.sin(_pitch / 2), 0.0)

SHELF_COLOR = (0.75, 0.75, 0.78)
PANEL_COLOR = (0.55, 0.55, 0.60)


def static_box(prim_name: str, size: tuple, pos: tuple, color: tuple) -> AssetBaseCfg:
    """A fixed box that the robot and objects can collide with (it never moves)."""
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
    """One horizontal shelf board. side = +1 (left) or -1 (right)."""
    y = side * (AISLE_HALF_WIDTH + SHELF_DEPTH / 2)
    name = f"{'left' if side > 0 else 'right'}_board_{level}"
    return static_box(
        name,
        size=(SHELF_LENGTH, SHELF_DEPTH, BOARD_THICKNESS),
        pos=(SHELF_CENTER_X, y, SHELF_HEIGHTS[level]),
        color=SHELF_COLOR,
    )


def back_panel(side: int) -> AssetBaseCfg:
    y = side * (AISLE_HALF_WIDTH + SHELF_DEPTH + 0.01)
    name = f"{'left' if side > 0 else 'right'}_back_panel"
    return static_box(
        name,
        size=(SHELF_LENGTH, 0.02, UNIT_HEIGHT),
        pos=(SHELF_CENTER_X, y, UNIT_HEIGHT / 2),
        color=PANEL_COLOR,
    )


def shelf_top(level: int) -> float:
    """Height of the top surface of a shelf board."""
    return SHELF_HEIGHTS[level] + BOARD_THICKNESS / 2


# Banana on the left shelf, middle level, near the front edge so the arm can reach it.
# It is static for this demo so the detected world point can be used after it leaves view.
BANANA_SHELF_LEVEL = 1  # 0 = bottom (0.40 m), 1 = middle (0.80 m), 2 = top (1.10 m)
BANANA_POS = (2.0, AISLE_HALF_WIDTH + 0.10, shelf_top(BANANA_SHELF_LEVEL) + 0.022)


# Grocery products from NVIDIA's asset library (YCB objects with physics already set up).
# They are downloaded from NVIDIA's servers the first time they load.
YCB_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"


def grocery_item(name: str, usd_file: str, side: int, level: int, x: float) -> RigidObjectCfg:
    """A YCB product placed on a shelf. It is spawned slightly above the board and settles onto it."""
    y = side * (AISLE_HALF_WIDTH + 0.12)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.UsdFileCfg(usd_path=f"{YCB_DIR}/{usd_file}"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, shelf_top(level) + 0.12)),
    )


@configclass
class AisleSceneCfg(RobotSceneCfg):
    """The walking scene (robot, sensors, lights, floor) plus shelves and a target object."""

    # Head RGB-D camera, at the real G1's D435 mount (from Unitree's URDF: d435_joint on torso_link).
    # Cameras only render when the app is launched with --enable_cameras.
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
        update_period=0.1,
        height=720,
        width=960,
        data_types=["rgb", "distance_to_image_plane"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.2,  # ~69 deg horizontal field of view, like the D435 colour camera
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0576, 0.0175, 0.4299),
            rot=CAMERA_ROT,
            convention="world",  # camera looks along +x (forward), +z up
        ),
    )

    # left shelf unit (y > 0)
    left_board_0 = board(+1, 0)
    left_board_1 = board(+1, 1)
    left_board_2 = board(+1, 2)
    left_back_panel = back_panel(+1)

    # right shelf unit (y < 0)
    right_board_0 = board(-1, 0)
    right_board_1 = board(-1, 1)
    right_board_2 = board(-1, 2)
    right_back_panel = back_panel(-1)

    target_object = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetBanana",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/011_banana.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BANANA_POS,
            rot=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),  # +90 deg yaw, wxyz
        ),
    )

    # other products on the shelves (distractors for perception) - switched off for now, banana only
    # cracker_box = grocery_item("CrackerBox", "003_cracker_box.usd", side=+1, level=1, x=1.5)
    # sugar_box = grocery_item("SugarBox", "004_sugar_box.usd", side=+1, level=0, x=2.3)
    # soup_can = grocery_item("SoupCan", "005_tomato_soup_can.usd", side=-1, level=1, x=1.8)
    # mustard = grocery_item("Mustard", "006_mustard_bottle.usd", side=-1, level=2, x=2.4)

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.TargetVelocityCommandCfg(
        asset_name="robot",
        initial_radius_range=(2.0, 5.0),
        target_angle_range=(-math.pi, math.pi),
        yaw_gain=1.0,
        stop_distance=0.3,
        resampling_time_range=(1.0e9, 1.0e9),  # One stationary target per episode.
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        robot_pose = ObsTerm(func=mdp.robot_pose)
        target_pose = ObsTerm(func=mdp.target_pose)
        obstacle_states = ObsTerm(func=mdp.obstacle_states_placeholder, history_length=6)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            # A group override would replace the obstacle term's six-step history.
            self.history_length = 0
            for name in (
                "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                "joint_pos_rel", "joint_vel_rel", "last_action", "robot_pose", "target_pose",
            ):
                getattr(self, name).history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        robot_pose = ObsTerm(func=mdp.robot_pose)
        target_pose = ObsTerm(func=mdp.target_pose)
        obstacle_states = ObsTerm(func=mdp.obstacle_states_placeholder, history_length=6)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 0
            for name in (
                "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                "joint_pos_rel", "joint_vel_rel", "last_action", "robot_pose", "target_pose",
            ):
                getattr(self, name).history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    target_distance_debug = RewTerm(
        func=mdp.target_distance_debug, weight=1.0, params={"command_name": "base_velocity"}
    )

    # -- task
    target_distance = RewTerm(func=mdp.target_distance_reward, weight=3.0, params={"beta": 0.1})
    target_stop = RewTerm(func=mdp.target_stop_reward, weight=2.0, params={"std": 0.1})
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "use_target_distance": True,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = None
    lin_vel_cmd_levels = None
    target_radius_levels = None


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

    def enable_aisle(self):
        """Use the grocery aisle and wait for an RGB-D target during playback."""
        self.scene = AisleSceneCfg(
            num_envs=1,
            env_spacing=5.0,
            terrain=self.scene.terrain,
            robot=self.scene.robot,
            height_scanner=self.scene.height_scanner,
            contact_forces=self.scene.contact_forces,
            sky_light=self.scene.sky_light,
        )
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.external_target = True
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.episode_length_s = 120.0
