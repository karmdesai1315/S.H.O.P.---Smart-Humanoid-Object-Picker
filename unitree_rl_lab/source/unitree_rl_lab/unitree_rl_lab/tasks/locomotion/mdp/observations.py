from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Root XYZ relative to env origin, then world orientation quaternion (w,x,y,z)."""
    robot = env.scene["robot"]
    return torch.cat(
        (robot.data.root_pos_w - env.scene.env_origins, robot.data.root_quat_w), dim=-1
    )


def target_pose(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Target XYZ relative to env origin, then world quaternion (w,x,y,z).

    The current target is a ground-plane point with identity orientation. Height
    and orientation reserve pose fields; neither changes the XY walking command.
    """
    command = env.command_manager.get_term(command_name)
    pose = torch.zeros(env.num_envs, 7, device=env.device)
    pose[:, :2] = command.target_pos_w - env.scene.env_origins[:, :2]
    pose[:, 2] = command.cfg.target_height
    pose[:, 3:] = torch.tensor(command.cfg.target_orientation, device=env.device)
    return pose


def obstacle_states_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Three absent obstacle slots: [x,y,z,vx,vy,vz] per slot, all zeros.

    Reserved convention: positions relative to environment origin; velocities
    in world axes, in metres and metres/second. ObservationManager supplies the
    six-step history and clears histories on environment resets.
    """
    return torch.zeros(env.num_envs, 3 * 6, device=env.device)


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
