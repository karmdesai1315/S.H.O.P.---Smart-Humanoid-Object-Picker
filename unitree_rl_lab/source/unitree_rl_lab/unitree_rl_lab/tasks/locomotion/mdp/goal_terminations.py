"""Termination conditions for goal-conditioned G1 walking.

Success uses the final floor-level walking goal, not the target's height
or an intermediate waypoint.

Speed thresholds should match those used in goal_rewards.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GoalReached(ManagerTermBase):
    """Succeed after remaining inside the goal and nearly stationary.

    The termination manager calls this once per environment step.
    Per-environment counters are cleared on episode reset and whenever
    a new goal command is sampled.
    """

    def __init__(
        self,
        cfg: TerminationTermCfg,
        env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)

        self._settled_steps = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

        self._last_goal_counter = torch.full(
            (env.num_envs,),
            -1,
            dtype=torch.long,
            device=env.device,
        )

    def reset(
        self,
        env_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """Clear only the environments being reset."""
        if env_ids is None:
            env_ids = slice(None)

        self._settled_steps[env_ids] = 0
        self._last_goal_counter[env_ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "goal_position",
        linear_speed_threshold: float = 0.08,
        angular_speed_threshold: float = 0.10,
        settle_time_s: float = 0.5,
    ) -> torch.Tensor:
        """Return a success boolean for each environment.

        All conditions must hold continuously:
            distance_xy <= configured goal_radius
            horizontal speed <= linear_speed_threshold [m/s]
            absolute world-Z angular speed <= angular_speed_threshold
                [rad/s]

        Leaving the circle or exceeding either speed threshold resets
        that environment's settling counter.
        """
        if settle_time_s <= 0.0:
            raise ValueError("settle_time_s must be positive.")

        if linear_speed_threshold < 0.0 or angular_speed_threshold < 0.0:
            raise ValueError("Speed thresholds must be non-negative.")

        term = env.command_manager.get_term(command_name)

        # Do not carry settling time from one sampled goal to another.
        changed_goal = term.command_counter != self._last_goal_counter
        self._settled_steps[changed_goal] = 0
        self._last_goal_counter[:] = term.command_counter

        inside_goal = term.goal_distance_xy <= term.cfg.goal_radius

        horizontal_speed = torch.linalg.vector_norm(
            term.robot.data.root_lin_vel_w[:, :2],
            dim=-1,
        )
        yaw_speed = torch.abs(
            term.robot.data.root_ang_vel_w[:, 2]
        )

        settled = (
            inside_goal
            & (horizontal_speed <= linear_speed_threshold)
            & (yaw_speed <= angular_speed_threshold)
        )

        self._settled_steps[:] = torch.where(
            settled,
            self._settled_steps + 1,
            torch.zeros_like(self._settled_steps),
        )

        required_steps = max(
            1,
            math.ceil(settle_time_s / env.step_dt),
        )

        return self._settled_steps >= required_steps


def goal_timeout(
    env: ManagerBasedRLEnv,
    timeout_s: float = 20.0,
) -> torch.Tensor:
    """End a goal attempt after its episode time budget.

    Stage 1 uses one goal per episode, so the episode clock is also the
    goal-attempt clock. This does not restart when a waypoint changes.

    Register this with time_out=True so the learning algorithm treats
    it as a time-limit truncation.
    """
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive.")

    maximum_steps = max(
        1,
        math.ceil(timeout_s / env.step_dt),
    )

    return env.episode_length_buf >= maximum_steps