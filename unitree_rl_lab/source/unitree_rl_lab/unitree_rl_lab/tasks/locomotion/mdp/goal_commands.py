"""Goal commands and observations for goal-conditioned G1 walking.

Stage 1:
    The local waypoint is the final floor-level walking goal.

Stage 2:
    A separate obstacle planner can update waypoint_pos_w while keeping target_pos_w and goal_pos_w fixed.

Coordinate conventions:
    *_w: world coordinates, including each environment's origin.
    *_b: robot-centered coordinates rotated by inverse robot yaw.
         X is forward, Y is left, and Z remains vertical.

Command layout (6 values):
    [waypoint_x_b, waypoint_y_b,
     target_x_b, target_y_b, target_z_b,
     hold_position]

No target orientation or heading error is included.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GoalPositionCommand(CommandTerm):
    """Maintain a target, a walking goal, and a local waypoint per robot."""

    cfg: GoalPositionCommandCfg

    def __init__(
        self,
        cfg: GoalPositionCommandCfg,
        env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)

        if not 0.0 <= cfg.distance_range[0] <= cfg.distance_range[1]:
            raise ValueError("distance_range must satisfy 0 <= minimum <= maximum.")

        if cfg.target_height_range[0] > cfg.target_height_range[1]:
            raise ValueError("target_height_range minimum must not exceed maximum.")

        if not 0.0 < cfg.goal_radius < cfg.hold_release_radius:
            raise ValueError(
                "Expected 0 < goal_radius < hold_release_radius."
            )

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Target object position, including height.
        self.target_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )

        # Final floor-level approach position used for walking success.
        # In Stage 1, its XY matches the target's XY.
        self.goal_pos_w = torch.zeros_like(self.target_pos_w)

        # Immediate walking destination. Equals goal_pos_w in Stage 1.
        self.waypoint_pos_w = torch.zeros_like(self.target_pos_w)

        # Automatic goal-circle stop request, with hysteresis.
        self.goal_hold = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Separate stop request reserved for the obstacle planner.
        # It remains False throughout ordinary Stage 1 training.
        self.planner_hold = torch.zeros_like(self.goal_hold)

        self._command = torch.zeros(
            self.num_envs, 6, device=self.device
        )

        self.metrics["goal_distance_xy"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["inside_goal"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        """Six navigation observations, shape (num_envs, 6)."""
        return self._command

    @property
    def goal_distance_xy(self) -> torch.Tensor:
        """Current horizontal distance to the final walking goal.

        Compute from live robot state so rewards do not depend on when
        the command manager last refreshed its observation buffer.
        """
        delta = self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        return torch.linalg.vector_norm(delta, dim=-1)

    @property
    def hold_position(self) -> torch.Tensor:
        """Whether either goal logic or the planner requests a stop."""
        return self.goal_hold | self.planner_hold

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new target relative to each robot's current position."""
        count = len(env_ids)
        if count == 0:
            return

        robot_pos = self.robot.data.root_pos_w[env_ids]
        origins = self._env.scene.env_origins[env_ids]

        # Uniform sampling over the area of an annulus.
        minimum, maximum = self.cfg.distance_range
        radius = torch.sqrt(
            torch.rand(count, device=self.device)
            * (maximum**2 - minimum**2)
            + minimum**2
        )

        # Full range of world directions. No target heading is sampled.
        angle = (
            torch.rand(count, device=self.device) * 2.0 * math.pi
        )

        goal = robot_pos.clone()
        goal[:, 0] += radius * torch.cos(angle)
        goal[:, 1] += radius * torch.sin(angle)

        # Stage 1 assumes a flat floor at this height relative to env origin.
        goal[:, 2] = origins[:, 2] + self.cfg.floor_height

        self.goal_pos_w[env_ids] = goal
        self.waypoint_pos_w[env_ids] = goal

        target = goal.clone()
        height_min, height_max = self.cfg.target_height_range
        target[:, 2] += (
            height_min
            + torch.rand(count, device=self.device)
            * (height_max - height_min)
        )
        self.target_pos_w[env_ids] = target

        self.goal_hold[env_ids] = False
        self.planner_hold[env_ids] = False

        # Refresh after sampling so reset observations have the new goal.
        self._update_command()

    def _update_command(self):
        """Update relative positions and the automatic stop request."""
        distance = self.goal_distance_xy

        # Enter hold at 0.5 m. Release only after drifting beyond the
        # larger release radius, preventing rapid stop/walk switching.
        self.goal_hold[:] = torch.where(
            self.goal_hold,
            distance <= self.cfg.hold_release_radius,
            distance <= self.cfg.goal_radius,
        )

        robot_pos = self.robot.data.root_pos_w
        robot_yaw = yaw_quat(self.robot.data.root_quat_w)

        waypoint_b = quat_apply_inverse(
            robot_yaw,
            self.waypoint_pos_w - robot_pos,
        )
        target_b = quat_apply_inverse(
            robot_yaw,
            self.target_pos_w - robot_pos,
        )

        self._command[:, 0:2] = waypoint_b[:, :2]
        self._command[:, 2:5] = target_b
        self._command[:, 5] = self.hold_position.to(
            dtype=self._command.dtype
        )

    def _update_metrics(self):
        """Log distance and circle occupancy, not stationary success."""
        distance = self.goal_distance_xy
        self.metrics["goal_distance_xy"][:] = distance
        self.metrics["inside_goal"][:] = (
            distance <= self.cfg.goal_radius
        ).to(dtype=distance.dtype)


@configclass
class GoalPositionCommandCfg(CommandTermCfg):
    """Configuration for Stage 1 goal sampling."""

    class_type: type = GoalPositionCommand

    asset_name: str = "robot"

    # Keep the goal fixed within an episode. Episode reset samples
    # another goal. Later, the environment config will set episode length.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # Initial nearby goals; a later curriculum can increase this range.
    distance_range: tuple[float, float] = (0.75, 2.0)

    # Target height above the floor, independent of the walking goal.
    # Varying height is observation variation, not a reaching objective.
    target_height_range: tuple[float, float] = (0.6, 1.2)

    # Flat-floor height relative to each environment's origin.
    floor_height: float = 0.0

    goal_radius: float = 0.5
    hold_release_radius: float = 0.6

    # Goal visualization will be added in the playback/configuration step.
    debug_vis: bool = False


# ---------------------------------------------------------------------
# Observation functions.
# These return raw values; scaling/history belong in the environment cfg.
# ---------------------------------------------------------------------

def local_waypoint_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Local waypoint XY in the robot's yaw-aligned frame: (N, 2)."""
    return env.command_manager.get_command(command_name)[:, 0:2]


def target_position_xyz(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Target XYZ relative to the robot root, yaw-aligned: (N, 3)."""
    return env.command_manager.get_command(command_name)[:, 2:5]


def hold_position_observation(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_position",
) -> torch.Tensor:
    """Automatic stop request as a float observation: (N, 1)."""
    return env.command_manager.get_command(command_name)[:, 5:6]


def base_linear_velocity_xy(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Horizontal robot velocity in the yaw-aligned frame: (N, 2)."""
    robot: Articulation = env.scene[asset_name]
    velocity = quat_apply_inverse(
        yaw_quat(robot.data.root_quat_w),
        robot.data.root_lin_vel_w,
    )
    return velocity[:, :2]