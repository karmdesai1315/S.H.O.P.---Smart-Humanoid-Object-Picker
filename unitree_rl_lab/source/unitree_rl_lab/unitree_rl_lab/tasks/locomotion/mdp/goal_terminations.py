"""Termination functions for the S.H.O.P. goal-walking task (all stages).

File: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/goal_terminations.py

Stage 1 terminations (names used in aisle_env_cfg.py; fall_penalty in goal_rewards.py reads the last two):
  time_out          episode time is up (20 s)            -> time_out=True (not a failure, PPO bootstraps)
  base_height       pelvis below 0.2 m                   -> fell
  bad_orientation   tilted more than 0.8 rad (~46 deg)   -> fell

Reaching the goal does NOT end the episode in Stage 1. The robot walks in and keeps standing there until
the time-out, because standing at the goal pays the most reward per step. Ending early would throw that
away, and the policy would learn to never quite settle.
Success is logged instead: Metrics/goal_position/inside_goal = fraction of robots inside the goal circle
when their episode ended (shown in the terminal and W&B).

GoalReached (Akanksh's "inside the circle and settled for 0.5 s") is kept for later stages, e.g. when
reaching the goal should hand over to grasping. It is NOT in the Stage 1 terminations list.
If it is used later, give the success a big enough bonus to beat the reward the robot would have kept
collecting by staying (see the note in goal_rewards.py).

goal_timeout is gone: time_out with episode_length_s = 20 does the same thing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ===========================================================================
# Stage 1
# ===========================================================================
def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode length reached. Register with time_out=True."""
    return env.episode_length_buf >= env.max_episode_length


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Pelvis dropped below minimum_height (m): the robot fell."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2] < minimum_height


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Body tilted more than limit_angle (rad) from upright: the robot is falling."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.acos(-robot.data.projected_gravity_b[:, 2]).abs() > limit_angle


# ===========================================================================
# Later stages (not used in Stage 1)
# ===========================================================================
class GoalReached(ManagerTermBase):
    """True once the robot has stayed inside the goal circle, nearly still, for settle_time_s.

    Use in the env cfg as:
        goal_reached = DoneTerm(func=mdp.GoalReached, params={"command_name": "goal_position",
                                "lin_speed_max": 0.08, "yaw_speed_max": 0.10, "settle_time_s": 0.5})
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.settled_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.last_goal_counter = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.settled_steps[env_ids] = 0
        self.last_goal_counter[env_ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "goal_position",
        lin_speed_max: float = 0.08,
        yaw_speed_max: float = 0.10,
        settle_time_s: float = 0.5,
    ) -> torch.Tensor:
        term = env.command_manager.get_term(command_name)

        # a new goal restarts the settle timer
        new_goal = term.command_counter != self.last_goal_counter
        self.settled_steps[new_goal] = 0
        self.last_goal_counter[:] = term.command_counter

        inside = term.goal_distance_xy <= term.cfg.goal_radius
        lin_speed = torch.norm(term.robot.data.root_lin_vel_w[:, :2], dim=-1)
        yaw_speed = torch.abs(term.robot.data.root_ang_vel_w[:, 2])
        settled = inside & (lin_speed <= lin_speed_max) & (yaw_speed <= yaw_speed_max)

        # count settled steps in a row, reset to 0 as soon as it moves or leaves the circle
        self.settled_steps[:] = torch.where(settled, self.settled_steps + 1, torch.zeros_like(self.settled_steps))
        need = max(1, math.ceil(settle_time_s / env.step_dt))
        return self.settled_steps >= need