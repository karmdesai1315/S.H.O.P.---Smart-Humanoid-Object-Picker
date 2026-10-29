"""Grocery-aisle scene for the Unitree G1 walking policy.

Place this file next to velocity_env_cfg.py:
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/aisle_env_cfg.py

It reuses the walking task unchanged (same observations and actions, so the
trained policy still loads) and only changes the world:
  - flat floor instead of the cobblestone terrain
  - two shelf units facing each other, three shelves each
  - an apple (the target) on the left middle shelf, plus YCB grocery products
  - a head RGB-D camera at the real G1's D435 mount (run with --enable_cameras)
  - no random pushes, fixed start pose, long episodes

Layout (seen from above, robot starts at the origin facing +x):

        y
        ^      back panel  y = +1.10
        |   [==== left shelf unit ====]   y = +0.70 .. +1.10
  robot o-->   aisle (1.4 m wide)          x = 1.0 .. 3.0
        |   [==== right shelf unit ===]   y = -0.70 .. -1.10
                back panel  y = -1.10
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg

from .velocity_env_cfg import RobotPlayEnvCfg, RobotSceneCfg
import math


# ---------------------------------------------------------------------------
# Aisle dimensions (metres). Change these to reshape the aisle.
# ---------------------------------------------------------------------------
AISLE_HALF_WIDTH = 0.70  # distance from aisle centre line to shelf front edge
SHELF_DEPTH = 0.40
SHELF_LENGTH = 2.00
SHELF_CENTER_X = 2.00  # shelves run from x = 1.0 to x = 3.0
BOARD_THICKNESS = 0.03
SHELF_HEIGHTS = (0.40, 0.80, 1.10)  # top surface of each board is height + thickness/2
UNIT_HEIGHT = 1.40  # height of the back panel

# Head camera downward tilt in degrees (0 = straight ahead). The real G1's D435 is 47.6.
CAMERA_PITCH_DEG = 20.0 #0 straight, 20 natural, 30, lower to middle shelves, 47.6 actual g1
_pitch = math.radians(CAMERA_PITCH_DEG)
CAMERA_ROT = (math.cos(_pitch / 2), 0.0, math.sin(_pitch / 2), 0.0)

SHELF_COLOR = (0.75, 0.75, 0.78)
PANEL_COLOR = (0.55, 0.55, 0.60)
TARGET_COLOR = (0.85, 0.20, 0.15)


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


# Target object: an "apple" (red sphere, 8 cm across) on the left middle shelf,
# near the front edge so it is within reach from the aisle.
# NVIDIA's asset library has no apple model, so it is a simple sphere with apple-like size and mass.
APPLE_RADIUS = 0.04
APPLE_LEVEL = 1
APPLE_POS = (SHELF_CENTER_X, AISLE_HALF_WIDTH + 0.10, shelf_top(APPLE_LEVEL) + APPLE_RADIUS + 0.005)


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
    # The URDF importer merges head_link and d435_link into torso_link, so the camera is attached there.
    # Cameras only render when the app is launched with --enable_cameras.
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.2,  # ~69 deg horizontal field of view, like the D435 colour camera
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0576, 0.0175, 0.4299),
            rot=CAMERA_ROT,  # pitched 47.6 deg down
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

    # the object to retrieve: a movable apple with mass, so it can be picked up later
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

    # other products on the shelves (distractors for perception, and a realistic-looking aisle)
    cracker_box = grocery_item("CrackerBox", "003_cracker_box.usd", side=+1, level=1, x=1.5)
    sugar_box = grocery_item("SugarBox", "004_sugar_box.usd", side=+1, level=0, x=2.3)
    soup_can = grocery_item("SoupCan", "005_tomato_soup_can.usd", side=-1, level=1, x=1.8)
    mustard = grocery_item("Mustard", "006_mustard_bottle.usd", side=-1, level=2, x=2.4)


@configclass
class AisleEnvCfg(RobotPlayEnvCfg):
    """The G1 walking task, placed in the grocery aisle."""

    scene: AisleSceneCfg = AisleSceneCfg(num_envs=1, env_spacing=5.0)

    def __post_init__(self):
        # sets up the walking task as Unitree's play config does
        super().__post_init__()

        # one robot (override with --num_envs); env_spacing keeps copies of the aisle apart
        self.scene.num_envs = 1

        # flat floor instead of the generated cobblestone terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None  # needs a generated terrain
        self.curriculum.lin_vel_cmd_levels = None

        # no random shoves; start at the origin facing down the aisle (+x)
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

        # long episodes so the scene doesn't reset every 20 s
        self.episode_length_s = 120.0