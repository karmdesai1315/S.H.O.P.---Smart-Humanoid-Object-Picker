# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configurations for velocity and goal-conditioned locomotion."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Original velocity-command walking configuration."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100

    experiment_name = ""
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class GoalPPORunnerCfg(BasePPORunnerCfg):
    """Goal-conditioned walking, initially trained from scratch.

    Use this same network structure when continuing into Stage 2.
    """

    # At a 0.02 s environment timestep, each rollout spans 0.64 s.
    num_steps_per_env = 32

    # Initial training budget, not an expected convergence time.
    max_iterations = 20000
    save_interval = 100

    # Keep goal-policy checkpoints separate from velocity-policy runs.
    experiment_name = "unitree_g1_29dof_goal"

    # Normalize inputs with different scales: joints, velocities, and goals.
    # Preserve these statistics when continuing training into Stage 2.
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        # Moderate exploration for initial learning from scratch.
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,

        # Maintain exploration while limiting excessive action noise.
        entropy_coef=0.005,

        num_learning_epochs=5,
        num_mini_batches=4,

        # More conservative starting rate than the original configuration.
        learning_rate=3.0e-4,
        schedule="adaptive",

        # Give delayed goal completion more influence than gamma=0.99.
        gamma=0.995,
        lam=0.95,

        desired_kl=0.01,
        max_grad_norm=1.0,
    )