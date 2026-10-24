"""Drive a trained G1 walking policy with the keyboard in Isaac Lab.

Usage (from the unitree_rl_lab folder):
    python scripts/rsl_rl/play_keyboard.py --task Unitree-G1-29dof-Velocity

Keys (click the Isaac Sim viewport first):
    Arrow Up / Down     walk forward / backward
    Arrow Left / Right  step left / right
    Z / X               turn left / right
    L                   stop (reset command to zero)
"""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Drive a trained RSL-RL policy with the keyboard.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--forward_speed", type=float, default=0.5, help="Forward/backward speed in m/s.")
parser.add_argument("--side_speed", type=float, default=0.3, help="Sideways speed in m/s.")
parser.add_argument("--turn_speed", type=float, default=0.2, help="Turning speed in rad/s.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    # one robot, driven only by the keyboard
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, entry_point_key="play_env_cfg_entry_point"
    )
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)  # never pick a random command
    cmd_cfg.rel_standing_envs = 0.0
    env_cfg.events.push_robot = None  # no random shoves
    env_cfg.episode_length_s = 1.0e6  # no automatic reset every 20 s
    # camera follows the robot
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    # find the checkpoint (latest run by default, or --load_run / --checkpoint)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.forward_speed,
            v_y_sensitivity=args_cli.side_speed,
            omega_z_sensitivity=args_cli.turn_speed,
            sim_device=env.unwrapped.device,
        )
    )
    print(keyboard)
    command_term = env.unwrapped.command_manager.get_term("base_velocity")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            command_term.vel_command_b[:] = keyboard.advance()  # keyboard replaces the random command
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        sleep_time = dt - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)  # run at real-time speed

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
