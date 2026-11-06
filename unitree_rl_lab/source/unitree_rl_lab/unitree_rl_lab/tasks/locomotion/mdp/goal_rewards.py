"""Navigation rewards for goal-conditioned G1 walking.

These functions return one value per environment. Reward weights are
configured separately in goal_env_cfg.py.

Distance is measured horizontally to the final walking goal.
Target height and intermediate waypoints do not determine task success.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .goal_commands import GoalPositionCommand


def _get_goal_term(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> GoalPositionCommand:
    """Get the goal command term, including its live robot state."""
    return cast(
        "GoalPositionCommand",
        env.command_manager.get_term(command_name),
    )


def goal_distance_and_stop_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_position",
    linear_speed_threshold: float = 0.08,
    angular_speed_threshold: float = 0.10,
) -> torch.Tensor:
    """Reward approaching the goal and stopping inside its circle.

    Outside the goal circle:
        reward = exp(-distance_xy**2)

    Inside the goal circle:
        reward = 1 if nearly stationary, otherwise 0

    Near-stationary means BOTH:
        horizontal linear speed <= linear_speed_threshold [m/s]
        absolute world-Z angular speed <= angular_speed_threshold [rad/s]

    Thresholds are initial tuning values, not guarantees of standing
    stability. Existing balance and fall-related terms remain necessary.
    """
    if linear_speed_threshold < 0.0 or angular_speed_threshold < 0.0:
        raise ValueError("Speed thresholds must be non-negative.")

    term = _get_goal_term(env, command_name)

    # Live distance avoids dependence on command observation refresh order.
    distance = term.goal_distance_xy
    inside_goal = distance <= term.cfg.goal_radius

    linear_speed = torch.linalg.vector_norm(
        term.robot.data.root_lin_vel_w[:, :2],
        dim=-1,
    )
    angular_speed = torch.abs(
        term.robot.data.root_ang_vel_w[:, 2]
    )

    nearly_stationary = (
        (linear_speed <= linear_speed_threshold)
        & (angular_speed <= angular_speed_threshold)
    )

    approach_reward = torch.exp(-torch.square(distance))
    stationary_reward = nearly_stationary.to(dtype=distance.dtype)

    return torch.where(
        inside_goal,
        stationary_reward,
        approach_reward,
    )


def motion_inside_goal_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_position",
    angular_weight: float = 0.25,
) -> torch.Tensor:
    """Return a positive movement cost inside the final goal circle.

    Assign this term a NEGATIVE reward weight in goal_env_cfg.py.

    Cost:
        horizontal_speed**2 + angular_weight * world_yaw_rate**2

    Outside the goal circle, the cost is zero.

    The cost remains continuous in speed, encouraging the robot to
    settle rather than merely remain just below the success thresholds.
    """
    if angular_weight < 0.0:
        raise ValueError("angular_weight must be non-negative.")

    term = _get_goal_term(env, command_name)

    inside_goal = term.goal_distance_xy <= term.cfg.goal_radius

    horizontal_speed_squared = torch.sum(
        torch.square(term.robot.data.root_lin_vel_w[:, :2]),
        dim=-1,
    )
    yaw_rate_squared = torch.square(
        term.robot.data.root_ang_vel_w[:, 2]
    )

    movement_cost = (
        horizontal_speed_squared
        + angular_weight * yaw_rate_squared
    )

    return torch.where(
        inside_goal,
        movement_cost,
        torch.zeros_like(movement_cost),
    )