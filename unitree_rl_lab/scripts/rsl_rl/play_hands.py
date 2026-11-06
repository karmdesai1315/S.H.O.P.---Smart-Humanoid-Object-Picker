"""Drive the G1 with Inspire hands through the aisle, and open/close the hands from the keyboard.

The walking policy controls the 29 body joints (as in play_keyboard.py). The finger
joints are not part of the policy: this script sets their targets directly.

Usage (from the unitree_rl_lab folder):
    python scripts/rsl_rl/play_hands.py --task Unitree-G1-29dof-Aisle-Inspire ^
        --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity/2026-10-15_19-20-23/model_5198.pt

Keys (click the Isaac Sim viewport first):
    Arrow Up / Down     walk forward / backward
    Arrow Left / Right  step left / right
    Z / X               turn left / right
    L                   stop walking
    C                   close the hand(s)
    O                   open the hand(s)

Options:
    --hand left|right|both   which hand(s) C/O control (default: both)
    --close_time 1.0         seconds to fully close or open
"""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Drive the G1 with Inspire hands and open/close them.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Aisle-Inspire", help="Name of the task.")
parser.add_argument("--forward_speed", type=float, default=0.5, help="Forward/backward speed in m/s.")
parser.add_argument("--side_speed", type=float, default=0.3, help="Sideways speed in m/s.")
parser.add_argument("--turn_speed", type=float, default=0.2, help="Turning speed in rad/s.")
parser.add_argument("--hand", type=str, default="both", choices=["left", "right", "both"], help="Hand(s) to control.")
parser.add_argument("--close_time", type=float, default=1.0, help="Seconds to go from fully open to fully closed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # the aisle scene has a head camera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.assets.robots.g1_inspire import G1_BODY_JOINTS, hand_joint_targets
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def check_body_joint_order(robot, run_dir: str):
    """Make sure the 29 body joints are in the same order the policy was trained with.

    The training run saved deploy.yaml, whose joint_ids_map gives each body joint's position
    in the training robot. If the orders differ, the policy would drive the wrong joints.
    """
    body_order_now = [n for n in robot.joint_names if n in set(G1_BODY_JOINTS)]
    deploy_path = None
    for root, _, files in os.walk(run_dir):
        if "deploy.yaml" in files:
            deploy_path = os.path.join(root, "deploy.yaml")
            break
    if deploy_path is None:
        print("[CHECK] deploy.yaml not found in the run folder; skipping the joint-order check.")
        return
    with open(deploy_path) as f:
        ids_map = yaml.safe_load(f)["joint_ids_map"]
    trained_order = [G1_BODY_JOINTS[sdk_index] for sdk_index in ids_map]  # Isaac joint k = SDK joint ids_map[k]
    if trained_order == body_order_now:
        print("[CHECK] Body joint order matches the trained policy. OK")
    else:
        raise RuntimeError(
            "Body joint order differs from the trained policy!\n"
            f"  trained: {trained_order}\n  now:     {body_order_now}"
        )


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, entry_point_key="play_env_cfg_entry_point")
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)  # never pick a random command
    cmd_cfg.rel_standing_envs = 0.0
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 1.0e6
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.5, 1.5, 1.6)  # closer view, to see the hands
    env_cfg.viewer.lookat = (0.3, 0.0, 0.8)

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

    robot = env.unwrapped.scene["robot"]
    print(f"[INFO] Robot has {robot.num_joints} joints (29 body + {robot.num_joints - 29} hand).")
    check_body_joint_order(robot, os.path.dirname(resume_path))

    # finger joints to drive, and their targets when fully open / fully closed
    sides = {"left": ["L"], "right": ["R"], "both": ["L", "R"]}[args_cli.hand]
    open_targets, closed_targets = {}, {}
    for s in sides:
        open_targets.update(hand_joint_targets(s, 0.0))
        closed_targets.update(hand_joint_targets(s, 1.0))
    hand_names = list(open_targets.keys())
    hand_ids = [robot.joint_names.index(n) for n in hand_names]
    q_open = torch.tensor([open_targets[n] for n in hand_names], device=robot.device).unsqueeze(0)
    q_closed = torch.tensor([closed_targets[n] for n in hand_names], device=robot.device).unsqueeze(0)
    print(f"[HANDS] Controlling {len(hand_ids)} finger joints: {hand_names}")

    # keyboard: walking (built in) + C/O for the hands
    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.forward_speed,
            v_y_sensitivity=args_cli.side_speed,
            omega_z_sensitivity=args_cli.turn_speed,
            sim_device=env.unwrapped.device,
        )
    )
    hand_state = {"goal": 0.0}  # 0 = open, 1 = closed

    def close_hands():
        hand_state["goal"] = 1.0
        print("[HANDS] closing")

    def open_hands():
        hand_state["goal"] = 0.0
        print("[HANDS] opening")

    keyboard.add_callback("C", close_hands)
    keyboard.add_callback("O", open_hands)
    print(keyboard)
    print("[HANDS] Press C to close, O to open.")
    command_term = env.unwrapped.command_manager.get_term("base_velocity")

    dt = env.unwrapped.step_dt
    closure = 0.0
    obs = env.get_observations()
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            # move the hand smoothly toward open/closed
            step = dt / max(args_cli.close_time, 1e-3)
            closure = min(hand_state["goal"], closure + step) if hand_state["goal"] > closure else max(hand_state["goal"], closure - step)
            robot.set_joint_position_target(q_open + closure * (q_closed - q_open), joint_ids=hand_ids)

            command_term.vel_command_b[:] = keyboard.advance()
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        sleep_time = dt - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
