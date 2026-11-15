"""Reward functions for the S.H.O.P. goal-walking task (all stages).

File: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/goal_rewards.py

Every reward the task uses is written out here (copied from Unitree's rewards.py and Isaac Lab's
locomotion rewards, same maths), so nothing has to be looked up in other files when debugging.
Weights are set in aisle_env_cfg.py.

Idea (Stage 1):
  - The walking rewards are Unitree's, unchanged. They track `base_velocity`, which is now the guide
    velocity from goal_commands.py (walk toward the goal, zero inside the goal circle). So the robot
    gets its usual walking reward for heading to the goal, and for standing still once it's there.
  - On top: a penalty for falling, and a small reward for every step spent standing still inside the circle.
  - Reaching the goal does not end the episode (see goal_terminations.py). Standing at the goal pays the
    most per step (~3 vs ~2.8 walking vs ~2.3 standing away from it), so the best plan is: get there fast, stay.

Note: Isaac Lab multiplies every weight by step_dt (0.02), so one-off terms need big weights
(fall -100 -> -2, about one second of reward).

Planned weights (Unitree's walking weights are unchanged):
  walking:  track_lin_vel_xy 1.0, track_ang_vel_z 0.5, alive 0.15, gait 0.5, feet_clearance 1.0
  penalties: lin_vel_z -2, ang_vel_xy -0.05, joint_vel -0.001, joint_acc -2.5e-7, action_rate -0.05,
             dof_pos_limits -5, energy -2e-5, joint_deviation_arms -0.1, waists -1, legs -1,
             flat_orientation -5, base_height -10, feet_slide -0.2, undesired_contacts -1
  goal:     fall -100 (once), at_goal_still +0.5 (per step)

If it still learns to stand still away from the goal, turn these first:
  track_lin_vel_xy 1.0 -> 1.5, guide min_speed 0.2 -> 0.3, gate feet_clearance on the guide command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Print goal debug info every N steps (env 0 + averages). 0 = off. Keep it off for full training runs.
DEBUG_EVERY = 0


# ===========================================================================
# Walking: velocity tracking (reads the base_velocity command = guide velocity)
# ===========================================================================
def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-error^2 / std^2) between commanded and actual (vx, vy), in the robot's heading frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-error^2 / std^2) between commanded and actual yaw rate."""
    robot: Articulation = env.scene[asset_cfg.name]
    error = torch.square(env.command_manager.get_command(command_name)[:, 2] - robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 every step the episode hasn't ended by a termination (time-outs don't count)."""
    return (~env.termination_manager.terminated).float()


# ===========================================================================
# Base penalties
# ===========================================================================
def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Bouncing up and down."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Rocking (roll / pitch rate)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body tilt (gravity not straight down in the body frame)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Pelvis height away from target_height (flat floor at z = 0)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_pos_w[:, 2] - target_height)


# ===========================================================================
# Joint penalties
# ===========================================================================
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Jerky actions (change from the previous action)."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """How far joints go past their soft limits."""
    robot: Articulation = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    below = -(q - limits[..., 0]).clip(max=0.0)
    above = (q - limits[..., 1]).clip(min=0.0)
    return torch.sum(below + above, dim=1)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """|joint speed| x |torque|, summed (Unitree)."""
    robot: Articulation = env.scene[asset_cfg.name]
    qvel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = robot.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """How far the selected joints are from their default pose (arms, waist, hip roll/yaw)."""
    robot: Articulation = env.scene[asset_cfg.name]
    diff = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(diff), dim=1)


# ===========================================================================
# Feet
# ===========================================================================
def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Unitree's gait clock: +1 per foot whose contact matches the planned stance/swing phase.
    Only counts while the command is above 0.1 (so it switches off when the guide says stop)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([(global_phase + o) % 1.0 for o in offset], dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str | None = None,
) -> torch.Tensor:
    """Unitree: reward moving feet for lifting to target_height.
    With command_name set, it only pays while that command is above 0.1 (like feet_gait),
    so standing still doesn't get it for free when the guide says walk."""
    robot: Articulation = env.scene[asset_cfg.name]
    z_error = torch.square(robot.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    swing = torch.tanh(tanh_mult * torch.norm(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.exp(-torch.sum(z_error * swing, dim=1) / std)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Feet sliding while in contact with the ground."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    robot: Articulation = env.scene[asset_cfg.name]
    feet_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Number of selected bodies (everything except the feet) touching something."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    is_contact = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1).float()


# ===========================================================================
# Goal rewards (on top of walking)
# ===========================================================================
def fall_penalty(env: ManagerBasedRLEnv, term_names: list[str] = ["base_height", "bad_orientation"]) -> torch.Tensor:
    """1 on the step the robot falls (any of the fall terminations). Give it a negative weight."""
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in term_names:
        fell |= env.termination_manager.get_term(name)
    return fell.float()


def at_goal_still(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_position",
    lin_speed_max: float = 0.1,
    yaw_speed_max: float = 0.15,
) -> torch.Tensor:
    """1 for every step inside the goal circle while nearly still, else 0."""
    term = env.command_manager.get_term(command_name)
    robot: Articulation = term.robot
    dist = term.goal_distance_xy
    inside = dist <= term.cfg.goal_radius
    lin_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    yaw_speed = torch.abs(robot.data.root_ang_vel_w[:, 2])
    still = (lin_speed <= lin_speed_max) & (yaw_speed <= yaw_speed_max)

    if DEBUG_EVERY > 0 and env.common_step_counter % DEBUG_EVERY == 0:
        guide = env.command_manager.get_command("base_velocity")
        print(
            f"[goal-debug] step={env.common_step_counter} "
            f"dist0={dist[0].item():.2f} dist_mean={dist.mean().item():.2f} "
            f"inside={inside.float().mean().item():.2f} still_inside={(inside & still).float().mean().item():.2f} "
            f"speed0={lin_speed[0].item():.2f} guide0=[{guide[0, 0].item():.2f}, {guide[0, 1].item():.2f}, "
            f"{guide[0, 2].item():.2f}]"
        )

    return (inside & still).float()