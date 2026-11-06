"""Register G1 29-DOF velocity and goal-conditioned walking tasks."""

import gymnasium as gym


# Original velocity-command walking task.
gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents."
            "rsl_rl_ppo_cfg:BasePPORunnerCfg"
        ),
    },
)


# Goal-conditioned walking task.
# Stage 1/Stage 2 scene settings are maintained in aisle_env_cfg.py.
gym.register(
    id="Unitree-G1-29dof-Aisle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.aisle_env_cfg:AisleEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.aisle_env_cfg:AislePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents."
            "rsl_rl_ppo_cfg:GoalPPORunnerCfg"
        ),
    },
)