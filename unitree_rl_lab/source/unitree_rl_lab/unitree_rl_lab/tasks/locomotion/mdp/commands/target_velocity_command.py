from __future__ import annotations

import torch

from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.utils import configclass

from .velocity_command import UniformLevelVelocityCommandCfg


class TargetVelocityCommand(UniformVelocityCommand):
    """Track a fixed XY target with velocity commands in the robot's yaw frame."""

    def _resample_command(self, env_ids):
        # Keep the same target through the episode; do not sample random velocities.
        self.vel_command_b[env_ids] = 0.0

    def _update_command(self):
        # Target coordinates are relative to each environment's world origin,
        # not the robot, so the target stays fixed as the robot moves.
        target_xy_w = self._env.scene.env_origins[:, :2] + torch.tensor(
            self.cfg.target_xy, device=self.device
        )
        error_w = target_xy_w - self.robot.data.root_pos_w[:, :2]

        # World XY error -> robot yaw frame (quaternions are w, x, y, z).
        w, x, y, z = self.robot.data.root_quat_w.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        error_b = torch.stack(
            (cos_yaw * error_w[:, 0] + sin_yaw * error_w[:, 1],
             -sin_yaw * error_w[:, 0] + cos_yaw * error_w[:, 1]),
            dim=-1,
        )
        velocity = self.cfg.position_gain * error_b

        # Scale both axes together to preserve the direction toward the target,
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
    target_xy: tuple[float, float] = (3.0, 0.0)  # metres, relative to each environment origin
    position_gain: float = 0.5  # 1/s
    stop_distance: float = 0.1  # metres
