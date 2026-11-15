"""Register the S.H.O.P. tasks.

File: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py

Only gymnasium is imported here (the task list is built before Isaac Sim starts).
Task ids must contain "Unitree", or train.py / play.py won't list them.
"""

import gymnasium as gym

# Stage 1: walk to a random goal on a flat floor and stand inside the circle.
gym.register(
    id="Unitree-G1-29dof-Shop-Stage1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aisle_env_cfg:Stage1EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.aisle_env_cfg:Stage1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:ShopPPORunnerCfg",
    },
)