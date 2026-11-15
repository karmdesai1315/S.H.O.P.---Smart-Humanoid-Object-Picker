"""Commands for the S.H.O.P. goal-walking task (all stages).

File: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/goal_commands.py

Three commands, defined in this order in the env cfg (order matters, base_velocity reads goal_position):

  goal_position   where to walk, where the object is, and when to stop.        policy sees it (12 values)
  base_velocity   guide velocity worked out from the goal (vx, vy, yaw rate).   policy does NOT see it
                  Only the walking rewards use it, so Unitree's walking rewards work unchanged.
  obstacles       K = 4 objects in front of the robot (6-DoF pose + size).     policy sees it (44 values)

Frames:
  *_w   world frame
  *_b   robot frame, yaw only (x forward, y left, z up), centred on the pelvis

goal_position layout (12):
  [0:2]   goal_xy_b       final walking spot (where to stand)
  [2:4]   waypoint_xy_b   next spot to walk to (= goal in Stage 1, obstacle planner changes it later)
  [4:7]   target_pos_b    gripper target position (the object)
  [7:11]  target_quat_b   gripper target orientation (w, x, y, z), relative to robot yaw
  [11]    hold            1 = stop here

obstacles layout (4 slots x 11 = 44), per slot:
  [pos_b (3), quat_b (4), size (3), valid (1)]
  Stage 1 has no obstacles: the slots hold random fake objects with valid = 0, so the policy learns to
  ignore valid = 0 and the observation normalizer already knows the value range when real ones arrive.
  Stage 2: call set_world_obstacles(...) with the scene objects; the 4 nearest ones in front fill the slots.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_inv, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def random_yaw_quat(n: int, device) -> torch.Tensor:
    """n random rotations about z, as (w, x, y, z)."""
    yaw = (torch.rand(n, device=device) * 2 - 1) * math.pi
    zero = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zero, zero, yaw)


def to_robot_frame(robot_pos_w, robot_yaw_q, points_w):
    """World points (N, M, 3) -> robot yaw frame (N, M, 3)."""
    n, m, _ = points_w.shape
    q = robot_yaw_q[:, None, :].expand(n, m, 4).reshape(-1, 4)
    v = (points_w - robot_pos_w[:, None, :]).reshape(-1, 3)
    return quat_apply_inverse(q, v).reshape(n, m, 3)


def quat_to_robot_frame(robot_yaw_q, quat_w):
    """World orientations (N, M, 4) -> relative to robot yaw (N, M, 4)."""
    n, m, _ = quat_w.shape
    q = robot_yaw_q[:, None, :].expand(n, m, 4).reshape(-1, 4)
    return quat_mul(quat_inv(q), quat_w.reshape(-1, 4)).reshape(n, m, 4)


def compute_guide_velocity(goal_xy_b: torch.Tensor, hold: torch.Tensor, cfg: GuideVelocityCommandCfg) -> torch.Tensor:
    """Velocity command (N, 3) = [vx, vy, yaw rate] that walks the robot to the goal.

    - speed grows with distance: k_lin * dist, between min_speed and max_speed
    - walks straight at the goal, so a goal to the side or behind gives some sideways / backward walking
      (clipped to max_side / max_backward)
    - turns toward the goal at k_turn * heading error (clipped to max_turn), except for goals closer than
      turn_min_dist, which it just steps to
    - all zero while holding (inside the goal circle)
    """
    dx, dy = goal_xy_b[:, 0], goal_xy_b[:, 1]
    dist = torch.sqrt(dx**2 + dy**2)
    heading_err = torch.atan2(dy, dx)

    speed = torch.clamp(cfg.k_lin * dist, cfg.min_speed, cfg.max_speed)
    vx = torch.clamp(speed * torch.cos(heading_err), -cfg.max_backward, cfg.max_forward)
    vy = torch.clamp(speed * torch.sin(heading_err), -cfg.max_side, cfg.max_side)
    wz = torch.clamp(cfg.k_turn * heading_err, -cfg.max_turn, cfg.max_turn)
    wz = torch.where(dist < cfg.turn_min_dist, torch.zeros_like(wz), wz)

    cmd = torch.stack([vx, vy, wz], dim=-1)
    cmd[hold] = 0.0
    return cmd


# ---------------------------------------------------------------------------
# 1) goal_position
# ---------------------------------------------------------------------------
class GoalPositionCommand(CommandTerm):
    """Walking goal, waypoint, gripper target pose and the stop (hold) flag, one per robot."""

    cfg: GoalPositionCommandCfg

    def __init__(self, cfg: GoalPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        n = self.num_envs
        self.goal_pos_w = torch.zeros(n, 3, device=self.device)  # where to stand (floor height)
        self.waypoint_pos_w = torch.zeros(n, 3, device=self.device)  # next spot to walk to
        self.target_pos_w = torch.zeros(n, 3, device=self.device)  # the object (gripper target)
        self.target_quat_w = torch.zeros(n, 4, device=self.device)
        self.target_quat_w[:, 0] = 1.0

        self.goal_hold = torch.zeros(n, dtype=torch.bool, device=self.device)  # inside the goal circle
        self.planner_hold = torch.zeros(n, dtype=torch.bool, device=self.device)  # for the obstacle planner later

        self._command = torch.zeros(n, 12, device=self.device)

        self.metrics["goal_distance"] = torch.zeros(n, device=self.device)
        self.metrics["inside_goal"] = torch.zeros(n, device=self.device)

    def __str__(self) -> str:
        return (
            f"GoalPositionCommand: goals {self.cfg.distance_range} m away, "
            f"goal radius {self.cfg.goal_radius} m (release {self.cfg.hold_release_radius} m)"
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def goal_distance_xy(self) -> torch.Tensor:
        """Horizontal distance to the goal, from the live robot state."""
        return torch.norm(self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=-1)

    @property
    def hold_position(self) -> torch.Tensor:
        return self.goal_hold | self.planner_hold

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        robot_pos = self.robot.data.root_pos_w[env_ids]
        origins = self._env.scene.env_origins[env_ids]

        # goal: random direction, distance uniform over the ring area
        lo, hi = self.cfg.distance_range
        r = torch.sqrt(torch.rand(n, device=self.device) * (hi**2 - lo**2) + lo**2)
        angle = torch.rand(n, device=self.device) * 2 * math.pi
        goal = robot_pos.clone()
        goal[:, 0] += r * torch.cos(angle)
        goal[:, 1] += r * torch.sin(angle)
        goal[:, 2] = origins[:, 2] + self.cfg.floor_height
        self.goal_pos_w[env_ids] = goal
        self.waypoint_pos_w[env_ids] = goal

        # gripper target: above the goal at a random height, random yaw (Stage 1 only uses it as an input)
        target = goal.clone()
        h_lo, h_hi = self.cfg.target_height_range
        target[:, 2] += h_lo + torch.rand(n, device=self.device) * (h_hi - h_lo)
        self.target_pos_w[env_ids] = target
        self.target_quat_w[env_ids] = random_yaw_quat(n, self.device)

        self.goal_hold[env_ids] = False
        self.planner_hold[env_ids] = False
        self._update_command()

    def _update_command(self):
        # hold switches on inside goal_radius and only switches off past hold_release_radius
        dist = self.goal_distance_xy
        self.goal_hold[:] = torch.where(
            self.goal_hold, dist <= self.cfg.hold_release_radius, dist <= self.cfg.goal_radius
        )

        robot_pos = self.robot.data.root_pos_w
        robot_yaw = yaw_quat(self.robot.data.root_quat_w)
        points_b = to_robot_frame(
            robot_pos, robot_yaw, torch.stack([self.goal_pos_w, self.waypoint_pos_w, self.target_pos_w], dim=1)
        )
        target_quat_b = quat_to_robot_frame(robot_yaw, self.target_quat_w[:, None, :])[:, 0]

        self._command[:, 0:2] = points_b[:, 0, :2]
        self._command[:, 2:4] = points_b[:, 1, :2]
        self._command[:, 4:7] = points_b[:, 2]
        self._command[:, 7:11] = target_quat_b
        self._command[:, 11] = self.hold_position.float()

    def _update_metrics(self):
        dist = self.goal_distance_xy
        self.metrics["goal_distance"][:] = dist
        self.metrics["inside_goal"][:] = (dist <= self.cfg.goal_radius).float()

    # green ball = goal, frame = gripper target pose
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_marker"):
                self.goal_marker = VisualizationMarkers(self.cfg.goal_marker_cfg)
                self.target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)
            self.goal_marker.set_visibility(True)
            self.target_marker.set_visibility(True)
        elif hasattr(self, "goal_marker"):
            self.goal_marker.set_visibility(False)
            self.target_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        goal = self.goal_pos_w.clone()
        goal[:, 2] += 0.05
        self.goal_marker.visualize(translations=goal)
        self.target_marker.visualize(translations=self.target_pos_w, orientations=self.target_quat_w)


@configclass
class GoalPositionCommandCfg(CommandTermCfg):
    class_type: type = GoalPositionCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # one goal per episode

    distance_range: tuple[float, float] = (0.5, 1.5)  # goal distance from the start (m); curriculum widens it
    target_height_range: tuple[float, float] = (0.6, 1.2)  # object height above the floor (m)
    floor_height: float = 0.0
    goal_radius: float = 0.4  # inside this = at the goal
    hold_release_radius: float = 0.5  # hold stays on until the robot drifts past this

    debug_vis: bool = False
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1))
            )
        },
    )
    target_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/target")

    def __post_init__(self):
        self.target_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


# ---------------------------------------------------------------------------
# 2) base_velocity (guide velocity, not observed)
# ---------------------------------------------------------------------------
class GuideVelocityCommand(CommandTerm):
    """[vx, vy, yaw rate] toward the goal, recomputed every step. Named base_velocity in the env cfg
    so the walking rewards (velocity tracking, gait, ...) work exactly like in Unitree's task."""

    cfg: GuideVelocityCommandCfg

    def __init__(self, cfg: GuideVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["guide_speed"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"GuideVelocityCommand: follows '{self.cfg.goal_command_name}', max speed {self.cfg.max_speed} m/s"

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _resample_command(self, env_ids: Sequence[int]):
        self._update_command()

    def _update_command(self):
        goal = self._env.command_manager.get_term(self.cfg.goal_command_name)
        goal_xy_b = goal.command[:, 2:4]  # walk to the waypoint (= goal in Stage 1)
        self.vel_command_b[:] = compute_guide_velocity(goal_xy_b, goal.hold_position, self.cfg)

    def _update_metrics(self):
        self.metrics["guide_speed"][:] = torch.norm(self.vel_command_b[:, :2], dim=-1)


@configclass
class GuideVelocityCommandCfg(CommandTermCfg):
    class_type: type = GuideVelocityCommand
    goal_command_name: str = "goal_position"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # recomputed every step anyway

    k_lin: float = 1.0  # speed = k_lin * distance ...
    min_speed: float = 0.2  # ... but at least this (so it never stalls just outside the circle)
    max_speed: float = 0.6  # ... and at most this
    max_forward: float = 0.6
    max_backward: float = 0.3
    max_side: float = 0.3
    k_turn: float = 1.0  # yaw rate = k_turn * heading error ...
    max_turn: float = 0.5  # ... capped (rad/s)
    turn_min_dist: float = 0.6  # closer than this: just step there, no turning

    @configclass
    class Ranges:
        """Only read by export_deploy_cfg (deploy.yaml). Keep in line with the limits above."""

        lin_vel_x: tuple[float, float] = (-0.3, 0.6)
        lin_vel_y: tuple[float, float] = (-0.3, 0.3)
        ang_vel_z: tuple[float, float] = (-0.5, 0.5)

    ranges: Ranges = Ranges()


# ---------------------------------------------------------------------------
# 3) obstacles
# ---------------------------------------------------------------------------
SLOT_SIZE = 11  # pos 3 + quat 4 + size 3 + valid 1


class ObstacleCommand(CommandTerm):
    """K obstacle slots in the robot frame: the nearest objects in front of the robot."""

    cfg: ObstacleCommandCfg

    def __init__(self, cfg: ObstacleCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        n, k = self.num_envs, cfg.num_slots

        # what the policy sees
        self.pos_b = torch.zeros(n, k, 3, device=self.device)
        self.quat_b = torch.zeros(n, k, 4, device=self.device)
        self.quat_b[..., 0] = 1.0
        self.size = torch.zeros(n, k, 3, device=self.device)
        self.valid = torch.zeros(n, k, device=self.device)

        # fake objects for empty slots (resampled every episode)
        self.fake_pos_b = torch.zeros(n, k, 3, device=self.device)
        self.fake_quat_b = self.quat_b.clone()
        self.fake_size = torch.zeros(n, k, 3, device=self.device)

        # real objects in the world, set with set_world_obstacles() (Stage 2+)
        self.world_pos = None  # (n, M, 3)
        self.world_quat = None  # (n, M, 4)
        self.world_size = None  # (n, M, 3)

        self._command = torch.zeros(n, k * SLOT_SIZE, device=self.device)
        self.metrics["num_obstacles"] = torch.zeros(n, device=self.device)

    def __str__(self) -> str:
        real = "none (fake slots)" if self.world_pos is None else f"{self.world_pos.shape[1]} objects"
        return f"ObstacleCommand: {self.cfg.num_slots} slots, world obstacles: {real}"

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def set_world_obstacles(self, pos_w: torch.Tensor, quat_w: torch.Tensor, size: torch.Tensor):
        """Give the obstacles in the scene: pos_w (n, M, 3), quat_w (n, M, 4) as (w, x, y, z), size (n, M, 3)."""
        self.world_pos, self.world_quat, self.world_size = pos_w, quat_w, size

    def clear_world_obstacles(self):
        self.world_pos = self.world_quat = self.world_size = None

    def _resample_command(self, env_ids: Sequence[int]):
        n, k = len(env_ids), self.cfg.num_slots
        if n == 0:
            return
        c = self.cfg
        u = torch.rand(n, k, 3, device=self.device)
        lo = torch.tensor([c.fake_x_range[0], c.fake_y_range[0], c.fake_z_range[0]], device=self.device)
        hi = torch.tensor([c.fake_x_range[1], c.fake_y_range[1], c.fake_z_range[1]], device=self.device)
        self.fake_pos_b[env_ids] = lo + u * (hi - lo)
        self.fake_quat_b[env_ids] = random_yaw_quat(n * k, self.device).reshape(n, k, 4)
        s_lo, s_hi = c.fake_size_range
        self.fake_size[env_ids] = s_lo + torch.rand(n, k, 3, device=self.device) * (s_hi - s_lo)
        self._update_command()

    def _update_command(self):
        # start from the fake objects, valid = 0
        self.pos_b[:] = self.fake_pos_b
        self.quat_b[:] = self.fake_quat_b
        self.size[:] = self.fake_size
        self.valid[:] = 0.0

        if self.world_pos is not None:
            robot_yaw = yaw_quat(self.robot.data.root_quat_w)
            pos_b = to_robot_frame(self.robot.data.root_pos_w, robot_yaw, self.world_pos)
            quat_b = quat_to_robot_frame(robot_yaw, self.world_quat)

            # keep objects in front and within range, take the k nearest
            dist = torch.norm(pos_b[..., :2], dim=-1)
            ok = (pos_b[..., 0] > self.cfg.min_forward) & (dist < self.cfg.max_range)
            dist = torch.where(ok, dist, torch.full_like(dist, 1.0e6))
            k = min(self.cfg.num_slots, dist.shape[1])
            near_dist, idx = torch.topk(dist, k, dim=1, largest=False)
            found = near_dist < 1.0e5  # (n, k)

            def pick(x):
                return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))

            m = found[..., None]
            self.pos_b[:, :k] = torch.where(m, pick(pos_b), self.pos_b[:, :k])
            self.quat_b[:, :k] = torch.where(m, pick(quat_b), self.quat_b[:, :k])
            self.size[:, :k] = torch.where(m, pick(self.world_size), self.size[:, :k])
            self.valid[:, :k] = found.float()

        self._command[:] = torch.cat(
            [self.pos_b, self.quat_b, self.size, self.valid[..., None]], dim=-1
        ).reshape(self.num_envs, -1)

    def _update_metrics(self):
        self.metrics["num_obstacles"][:] = self.valid.sum(dim=1)


@configclass
class ObstacleCommandCfg(CommandTermCfg):
    class_type: type = ObstacleCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # fake slots change once per episode

    num_slots: int = 4
    min_forward: float = -0.2  # "in front" = x_b above this (m)
    max_range: float = 3.0  # ignore objects further than this (m)

    # fake objects for empty slots, in the robot frame (m). z is from the pelvis (~0.78 m up),
    # so -0.8 .. 0.7 covers floor to ~1.5 m, like real shelf items
    fake_x_range: tuple[float, float] = (0.0, 3.0)
    fake_y_range: tuple[float, float] = (-1.5, 1.5)
    fake_z_range: tuple[float, float] = (-0.8, 0.7)
    fake_size_range: tuple[float, float] = (0.05, 1.0)