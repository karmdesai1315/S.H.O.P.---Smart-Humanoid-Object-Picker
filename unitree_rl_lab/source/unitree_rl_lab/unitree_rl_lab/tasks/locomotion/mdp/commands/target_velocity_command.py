from __future__ import annotations

import math

import torch

from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.utils import configclass

from .velocity_command import UniformLevelVelocityCommandCfg


class TargetVelocityCommand(UniformVelocityCommand):
    """Track a sampled or fixed XY target with commands in the robot's yaw frame."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.radius_increment <= 0:
            raise ValueError("radius_increment must be positive")
        for low, high in (cfg.initial_radius_range, cfg.final_radius_range):
            if not 0.0 <= low <= high <= 5.0:
                raise ValueError("Target radius ranges must satisfy 0 <= min <= max <= 5 metres")
        # Public target variable: world XY coordinates, one target per robot.
        self.target_pos_w = self.robot.data.root_pos_w[:, :2].clone()
        self.holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_xy_w = self.target_pos_w.clone()
        self.hold_yaw_w = torch.zeros_like(self.holding, dtype=self.target_pos_w.dtype)
        if cfg.external_target:
            self._target_received = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.radius_range = tuple(cfg.initial_radius_range)
        for name in (
            "target_error_x", "target_error_y", "target_distance",
            "target_yaw_error", "yaw_rate", "yaw_rate_command",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        if cfg.yaw_gain is not None:
            self.target_bearing_w = torch.zeros_like(self.metrics["target_distance"])

    @property
    def target_distance(self) -> torch.Tensor:
        """Current horizontal distance to the target for each robot."""
        return torch.linalg.vector_norm(self.target_pos_w - self.robot.data.root_pos_w[:, :2], dim=-1)

    def _root_yaw(self, env_ids):
        w, x, y, z = self.robot.data.root_quat_w[env_ids].unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _error_in_yaw_frame(error_w, yaw):
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        return torch.stack(
            (cos_yaw * error_w[:, 0] + sin_yaw * error_w[:, 1],
             -sin_yaw * error_w[:, 0] + cos_yaw * error_w[:, 1]),
            dim=-1,
        )

    def _target_heading(self, error_w, distance, yaw, env_ids, reset=False):
        bearing = torch.atan2(error_w[:, 1], error_w[:, 0])
        # Hold the last valid bearing when the target is too close to define a heading.
        fallback = self.target_bearing_w[env_ids] if self.cfg.yaw_gain is not None and not reset else yaw
        bearing = torch.where(distance > self.cfg.min_bearing_distance, bearing, fallback)
        yaw_error = (bearing - yaw + math.pi) % (2.0 * math.pi) - math.pi
        return bearing, yaw_error

    def update_target_metrics(self):
        """Record signed errors in the robot's yaw frame and rates about world Z."""
        env_ids = slice(None)
        error_w = self.target_pos_w - self.robot.data.root_pos_w[:, :2]
        distance = torch.linalg.vector_norm(error_w, dim=-1)
        yaw = self._root_yaw(env_ids)
        error_b = self._error_in_yaw_frame(error_w, yaw)
        _, yaw_error = self._target_heading(error_w, distance, yaw, env_ids)

        self.metrics["target_error_x"].copy_(error_b[:, 0])
        self.metrics["target_error_y"].copy_(error_b[:, 1])
        self.metrics["target_distance"].copy_(distance)
        self.metrics["target_yaw_error"].copy_(yaw_error)
        self.metrics["yaw_rate"].copy_(self.robot.data.root_ang_vel_w[:, 2])
        self.metrics["yaw_rate_command"].copy_(self.vel_command_b[:, 2])

    def _resample_command(self, env_ids):
        self.holding[env_ids] = False
        if self.cfg.external_target:
            self._target_received[env_ids] = False
            self.target_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids, :2]
        elif self.cfg.target_xy is not None:
            # Pinpoint target relative to each environment's origin.
            self.target_pos_w[env_ids] = self._env.scene.env_origins[env_ids, :2] + torch.tensor(
                self.cfg.target_xy, device=self.device
            )
        else:
            # Get robot's current heading
            yaw = self._root_yaw(env_ids)

            if self.cfg.fixed_forward_distance is None:
                radius_min, radius_max = self.radius_range
                radius = torch.empty(len(env_ids), device=self.device).uniform_(radius_min, radius_max)
                # Some targets start inside the stop zone, so the robot has to turn on the spot to face them.
                if self.cfg.near_target_fraction > 0.0:
                    near = torch.rand(len(env_ids), device=self.device) < self.cfg.near_target_fraction
                    near_radius = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.near_radius_range)
                    radius = torch.where(near, near_radius, radius)
                angle = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.target_angle_range)
                heading = yaw + angle
            else:
                radius = self.cfg.fixed_forward_distance
                heading = yaw

            # Transform robot-relative target into world coordinates
            offset = torch.stack((radius * torch.cos(heading), radius * torch.sin(heading)), dim=-1)
            # Sample around the robot's actual reset position, including reset jitter.
            self.target_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids, :2] + offset
        self.vel_command_b[env_ids] = 0.0
        if self.cfg.yaw_gain is not None:
            self._update_target_yaw(env_ids, reset=True)
            self.vel_command_b[env_ids, 0] = self.cfg.ranges.lin_vel_x[1]  # walk forward from the start
        elif self.cfg.fixed_forward_distance is not None:
            self.vel_command_b[env_ids, 0] = self.cfg.ranges.lin_vel_x[1]

    def set_target_xy(self, env_ids, target_xy_w: torch.Tensor) -> None:
        """Supply a detected ground-plane target in world coordinates."""
        if not self.cfg.external_target:
            raise RuntimeError("Enable external_target before setting a detected target")
        self.target_pos_w[env_ids] = target_xy_w
        self._target_received[env_ids] = True

    def hold_position(self, env_ids) -> None:
        """Latch the robot's pickup stance independently of the observed object."""
        self.holding[env_ids] = True
        self.hold_xy_w[env_ids] = self.robot.data.root_pos_w[env_ids, :2]
        self.hold_yaw_w[env_ids] = self._root_yaw(env_ids)
        self.vel_command_b[env_ids] = 0.0

    def _apply_hold_command(self):
        yaw = self._root_yaw(slice(None))
        error_xy = self._error_in_yaw_frame(self.hold_xy_w - self.robot.data.root_pos_w[:, :2], yaw)
        error_yaw = (self.hold_yaw_w - yaw + math.pi) % (2.0 * math.pi) - math.pi
        xy_command = (self.cfg.hold_xy_gain * error_xy).clamp(-0.12, 0.12)
        yaw_command = (self.cfg.hold_yaw_gain * error_yaw).clamp(-0.25, 0.25)
        hold_command = torch.cat((xy_command, yaw_command.unsqueeze(-1)), dim=-1)
        self.vel_command_b.copy_(torch.where(self.holding.unsqueeze(-1), hold_command, self.vel_command_b))

    def _update_target_yaw(self, env_ids, reset=False):
        error_w = self.target_pos_w[env_ids] - self.robot.data.root_pos_w[env_ids, :2]
        distance = torch.linalg.vector_norm(error_w, dim=-1)
        yaw = self._root_yaw(env_ids)
        bearing, yaw_error = self._target_heading(error_w, distance, yaw, env_ids, reset=reset)
        self.target_bearing_w[env_ids] = bearing

        yaw_error = yaw_error.sign() * (yaw_error.abs() - self.cfg.yaw_deadband).clamp_min(0.0)
        yaw_rate = self.cfg.yaw_gain * yaw_error
        self.vel_command_b[env_ids, 2] = yaw_rate.clamp(*self.cfg.ranges.ang_vel_z)

    def _update_command(self):
        if self.cfg.external_target:
            # Keep a stand target under the robot until perception provides a goal.
            waiting = ~self._target_received
            self.target_pos_w[waiting] = self.robot.data.root_pos_w[waiting, :2]
        if self.cfg.yaw_gain is not None:
            self._update_target_yaw(slice(None))
            # walk forward at lin_vel_x while turning toward the target, stop inside the stop zone
            reached = self.target_distance <= self.cfg.stop_distance
            self.vel_command_b[:, 0] = (~reached) * self.cfg.ranges.lin_vel_x[1]
            if self.cfg.external_target:
                self.vel_command_b[waiting] = 0.0
            self._apply_hold_command()
            return

        # Target stays fixed in world coordinates while the robot approaches.
        error_w = self.target_pos_w - self.robot.data.root_pos_w[:, :2]
        reached = torch.linalg.vector_norm(error_w, dim=-1) <= self.cfg.stop_distance
        if self.cfg.fixed_forward_distance is not None:
            self.vel_command_b[:, 0] = (~reached) * self.cfg.ranges.lin_vel_x[1]
            self._apply_hold_command()
            return

        error_b = self._error_in_yaw_frame(error_w, self._root_yaw(slice(None)))
        # Signed square keeps the command pointing toward the target.
        velocity = self.cfg.position_gain * error_b * error_b.abs()

        # Scale both axes together to preserve the quadratic command's direction,
        # while respecting the existing velocity ranges and their curriculum.
        ranges = self.cfg.ranges
        limits = torch.tensor(
            [[-ranges.lin_vel_x[0], ranges.lin_vel_x[1]],
             [-ranges.lin_vel_y[0], ranges.lin_vel_y[1]]], device=self.device
        )
        directional_limits = torch.where(velocity >= 0.0, limits[:, 1], limits[:, 0])
        scale = torch.clamp(
            (directional_limits / velocity.abs().clamp_min(1.0e-6)).amin(dim=-1), max=1.0
        )
        self.vel_command_b[:, :2] = velocity * scale.unsqueeze(-1)
        self.vel_command_b[:, 2] = 0.0  # Translate toward the target without commanding a turn.
        self.vel_command_b[reached] = 0.0
        self._apply_hold_command()


@configclass
class TargetVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = TargetVelocityCommand
    external_target: bool = False  # Wait for a world XY target from perception.
    target_xy: tuple[float, float] | None = None  # None=random; (x,y)=fixed relative to env origin
    fixed_forward_distance: float | None = None  # metres ahead at reset, with constant forward speed
    target_angle_range: tuple[float, float] = (-math.pi / 6, math.pi / 6)  # radians from reset heading
    yaw_gain: float | None = None  # Set to use yaw tracking with no XY velocity command.
    yaw_deadband: float = math.radians(5.0)
    min_bearing_distance: float = 0.02  # metres
    target_height: float = 0.0  # metres relative to env origin, for pose observation only
    target_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # world w,x,y,z
    initial_radius_range: tuple[float, float] = (2.0, 3.0)  # metres from robot at reset
    final_radius_range: tuple[float, float] = (4.0, 5.0)
    radius_increment: float = 0.5  # metres per successful curriculum update
    position_gain: float = 0.5  # 1/(m*s), multiplying signed squared position error
    hold_xy_gain: float = 1.2  # (m/s)/m during pickup
    hold_yaw_gain: float = 1.5  # (rad/s)/rad during pickup
    stop_distance: float = 0.2  # metres
    near_target_fraction: float = 0.0  # share of episodes whose target starts inside the stop zone (turn in place)
    near_radius_range: tuple[float, float] = (0.25, 0.45)  # metres, keep below stop_distance
