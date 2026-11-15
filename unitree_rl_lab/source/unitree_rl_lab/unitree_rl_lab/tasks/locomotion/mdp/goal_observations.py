"""Observation functions for the S.H.O.P. goal-walking task (all stages).

File: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/goal_observations.py

Everything the policy and critic see comes from this file (no imports from Unitree's or Isaac Lab's
observation code), so it's all in one place when debugging.

Policy observation (fixed for every stage, so later stages can keep training the same network):

  robot (proprioception)                     size
    base_ang_vel        gyro (rad/s)            3
    projected_gravity   tilt                    3
    joint_pos_rel       joint pos - default    29
    joint_vel_rel       joint vel              29
    last_action         previous action        29
  task
    goal_xy             where to stand          2
    waypoint_xy         next spot to walk to    2
    target_pos          gripper target pos      3
    target_quat         gripper target quat     4
    hold_flag           1 = stop here           1
    obstacle_slots      4 x 11                 44
                                              ---
                                              149 per frame

Not in the policy: the guide velocity (base_velocity) and the true base linear velocity
(the real G1 can't measure that well). The critic gets both, since it only runs in simulation.

The goal / obstacle functions just read the commands from goal_commands.py:
  goal_position command (12): [goal_xy 0:2, waypoint_xy 2:4, target_pos 4:7, target_quat 7:11, hold 11]
  obstacles command (44):     4 slots x [pos 3, quat 4, size 3, valid 1]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# robot (proprioception) - same maths as Isaac Lab's built-in versions
# ---------------------------------------------------------------------------
def base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base angular velocity in the base frame (gyro), (N, 3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base linear velocity in the base frame, (N, 3). Critic only."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_b


def projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity direction in the base frame (how tilted the robot is), (N, 3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.projected_gravity_b


def joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions minus the default pose, (N, num_joints)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities minus the default (zero), (N, num_joints)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel[:, asset_cfg.joint_ids] - robot.data.default_joint_vel[:, asset_cfg.joint_ids]


def last_action(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """The previous action (all action terms, or just one by name)."""
    if action_name is None:
        return env.action_manager.action
    return env.action_manager.get_term(action_name).raw_actions


def gait_phase(env: ManagerBasedRLEnv, period: float = 0.8) -> torch.Tensor:
    """Gait clock [sin, cos] from the episode time (Unitree's version), (N, 2). Not used in Stage 1."""
    phase = (env.episode_length_buf * env.step_dt) % period / period
    return torch.stack([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)


# ---------------------------------------------------------------------------
# task: goal, gripper target, obstacles
# ---------------------------------------------------------------------------
def goal_xy(env: ManagerBasedRLEnv, command_name: str = "goal_position") -> torch.Tensor:
    """Where to stand, in the robot frame (x forward, y left), (N, 2)."""
    return env.command_manager.get_command(command_name)[:, 0:2]


def waypoint_xy(env: ManagerBasedRLEnv, command_name: str = "goal_position") -> torch.Tensor:
    """Next spot to walk to, robot frame, (N, 2). Same as goal_xy in Stage 1."""
    return env.command_manager.get_command(command_name)[:, 2:4]


def target_pos(env: ManagerBasedRLEnv, command_name: str = "goal_position") -> torch.Tensor:
    """Gripper target position, robot frame (z from the pelvis), (N, 3)."""
    return env.command_manager.get_command(command_name)[:, 4:7]


def target_quat(env: ManagerBasedRLEnv, command_name: str = "goal_position") -> torch.Tensor:
    """Gripper target orientation (w, x, y, z) relative to the robot's heading, (N, 4)."""
    return env.command_manager.get_command(command_name)[:, 7:11]


def hold_flag(env: ManagerBasedRLEnv, command_name: str = "goal_position") -> torch.Tensor:
    """1 when the robot should stand still (inside the goal circle), else 0, (N, 1)."""
    return env.command_manager.get_command(command_name)[:, 11:12]


def obstacle_slots(env: ManagerBasedRLEnv, command_name: str = "obstacles") -> torch.Tensor:
    """K obstacle slots, each [pos 3, quat 4, size 3, valid 1], robot frame, (N, K * 11)."""
    return env.command_manager.get_command(command_name)


# ---------------------------------------------------------------------------
# critic only (privileged, simulation-only)
# ---------------------------------------------------------------------------
def guide_velocity(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """The guide velocity [vx, vy, yaw rate] the walking rewards track, (N, 3).
    The critic needs it to predict those rewards; the policy works from the goal instead."""
    return env.command_manager.get_command(command_name)


def goal_distance(env: ManagerBasedRLEnv, command_name: str = "goal_position") -> torch.Tensor:
    """Horizontal distance to the goal (m), (N, 1)."""
    return env.command_manager.get_term(command_name).goal_distance_xy.unsqueeze(-1)