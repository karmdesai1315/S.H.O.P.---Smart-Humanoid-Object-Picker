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
        self.radius_range = tuple(cfg.initial_radius_range)

    def _resample_command(self, env_ids):
        if self.cfg.target_xy is not None:
            # Pinpoint target relative to each environment's origin.
            self.target_pos_w[env_ids] = self._env.scene.env_origins[env_ids, :2] + torch.tensor(
                self.cfg.target_xy, device=self.device
            )
        else:
            radius_min, radius_max = self.radius_range
            radius = torch.empty(len(env_ids), device=self.device).uniform_(radius_min, radius_max)
            angle = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
            offset = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
            # Sample around the robot's actual reset position, including reset jitter.
            self.target_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids, :2] + offset
        self.vel_command_b[env_ids] = 0.0

    def _update_command(self):
        # Target stays fixed in world coordinates while the robot approaches.
        error_w = self.target_pos_w - self.robot.data.root_pos_w[:, :2]

        # World XY error -> robot yaw frame (quaternions are w, x, y, z).
        w, x, y, z = self.robot.data.root_quat_w.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        error_b = torch.stack(
            (cos_yaw * error_w[:, 0] + sin_yaw * error_w[:, 1],
             -sin_yaw * error_w[:, 0] + cos_yaw * error_w[:, 1]),
            dim=-1,
        )
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
        reached = torch.linalg.vector_norm(error_w, dim=-1) <= self.cfg.stop_distance
        self.vel_command_b[reached] = 0.0


@configclass
class TargetVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = TargetVelocityCommand
    target_xy: tuple[float, float] | None = None  # None=random; (x,y)=fixed relative to env origin
    target_height: float = 0.0  # metres relative to env origin, for pose observation only
    target_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # world w,x,y,z
    initial_radius_range: tuple[float, float] = (2.0, 3.0)  # metres from robot at reset
    final_radius_range: tuple[float, float] = (4.0, 5.0)
    radius_increment: float = 0.5  # metres per successful curriculum update
    position_gain: float = 0.5  # 1/(m*s), multiplying signed squared position error
    stop_distance: float = 0.2  # metres
