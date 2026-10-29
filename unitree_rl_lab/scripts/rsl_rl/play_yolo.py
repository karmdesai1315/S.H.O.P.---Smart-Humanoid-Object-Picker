"""Drive the trained G1 walking policy with the keyboard and run YOLO on the head camera.

Same as play_keyboard.py, plus object detection on the robot's head camera
(the `head_camera` defined in aisle_env_cfg.py).

Usage (from the unitree_rl_lab folder):
    python scripts/rsl_rl/play_yolo.py --task Unitree-G1-29dof-Aisle ^
        --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity/2026-10-15_19-20-23/model_5198.pt

Options:
    --targets bottle,banana   COCO class names to look for (comma-separated)
    --conf 0.25               minimum detection confidence
    --detect_every 10         run YOLO every N steps (10 steps = 0.2 s)
    --show                    open a window showing the camera image with boxes drawn
    --yolo_model yolov8n.pt   which YOLO weights to use (downloaded automatically the first time)

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

parser = argparse.ArgumentParser(description="Drive a trained RSL-RL policy with the keyboard and run YOLO.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Aisle", help="Name of the task.")
parser.add_argument("--forward_speed", type=float, default=0.5, help="Forward/backward speed in m/s.")
parser.add_argument("--side_speed", type=float, default=0.3, help="Sideways speed in m/s.")
parser.add_argument("--turn_speed", type=float, default=0.2, help="Turning speed in rad/s.")
parser.add_argument("--camera", type=str, default="head_camera", help="Camera name in the scene config.")
parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLO weights file.")
parser.add_argument("--targets", type=str, default="bottle,banana,apple", help="Comma-separated class names to look for.")
parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
parser.add_argument("--detect_every", type=int, default=10, help="Run YOLO every N environment steps.")
parser.add_argument("--show", action="store_true", default=False, help="Show the camera image with boxes in a window.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # the head camera only renders with this on

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import cv2
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from ultralytics import YOLO

import isaaclab_tasks  # noqa: F401
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def camera_frame_bgr(env, camera_name: str) -> np.ndarray:
    """Latest head-camera image of robot 0 as a BGR uint8 numpy array (the format YOLO/OpenCV expect)."""
    rgb = env.unwrapped.scene[camera_name].data.output["rgb"][0, ..., :3]  # (H, W, 3) on the GPU, RGB
    return np.ascontiguousarray(rgb.cpu().numpy().astype(np.uint8)[:, :, ::-1])


def detect(model, frame_bgr: np.ndarray, conf: float):
    """Run YOLO on one image. Returns the result object and a list of (class_name, confidence, box_xyxy)."""
    result = model.predict(frame_bgr, conf=conf, verbose=False)[0]
    found = []
    for cls_id, score, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
        found.append((model.names[int(cls_id)], score, [round(v) for v in box]))
    return result, found


def main():
    # one robot, driven only by the keyboard
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, entry_point_key="play_env_cfg_entry_point"
    )
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)  # never pick a random command
    cmd_cfg.rel_standing_envs = 0.0
    env_cfg.events.push_robot = None  # no random shoves
    env_cfg.episode_length_s = 1.0e6  # no automatic reset
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

    # YOLO (COCO classes: 'bottle', 'banana', 'apple', 'cup', ... see model.names)
    model = YOLO(args_cli.yolo_model)
    targets = {t.strip() for t in args_cli.targets.split(",") if t.strip()}
    unknown = targets - set(model.names.values())
    if unknown:
        print(f"[YOLO] Warning: {sorted(unknown)} are not classes this model knows.")
    print(f"[YOLO] Model {args_cli.yolo_model} loaded. Looking for: {sorted(targets)}")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    step = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            command_term.vel_command_b[:] = keyboard.advance()  # keyboard replaces the random command
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        step += 1

        if step % args_cli.detect_every == 0:
            frame = camera_frame_bgr(env, args_cli.camera)
            result, found = detect(model, frame, args_cli.conf)
            hits = [f for f in found if f[0] in targets]
            others = [f"{name} {score:.2f}" for name, score, _ in found if name not in targets]
            if hits:
                for name, score, box in hits:
                    print(f"[YOLO t={step * dt:6.1f}s] TARGET {name} ({score:.2f}) box={box}")
            else:
                print(f"[YOLO t={step * dt:6.1f}s] no target" + (f" | also saw: {', '.join(others)}" if others else ""))
            if args_cli.show:
                cv2.imshow("head_camera + YOLO", result.plot())
                cv2.waitKey(1)

        sleep_time = dt - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)  # run at real-time speed

    if args_cli.show:
        cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()