"""Walk the Stage 1 G1 policy to an RGB-D detection in the grocery aisle.

Run from unitree_rl_lab with --checkpoint /path/to/stage1.pt.
Use --show to display detections in a desktop session.
"""

import argparse
import math
from importlib.metadata import version

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target", default="banana", help="YOLO class to approach.")
parser.add_argument("--yolo_model", default="yolov8n-seg.pt", help="YOLO weights file.")
parser.add_argument("--conf", type=float, default=0.25, help="Minimum YOLO confidence.")
parser.add_argument("--detect_every", type=int, default=5, help="Steps between detections.")
parser.add_argument("--show", action="store_true", help="Show the head-camera detections.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint:
    parser.error("--checkpoint must point to the trained Stage 1 policy")
if args.detect_every < 1:
    parser.error("--detect_every must be at least 1")
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import cv2
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from ultralytics import YOLO

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

TASK = "Unitree-G1-29dof-Velocity"
TARGET_SAMPLES = 3
TARGET_SPREAD_M = 0.15
SETTLE_SECONDS = 0.5


def detection_xy(camera_data, result, max_depth=6.0):
    """Localize the highest-confidence detection with usable aligned depth."""
    if result.boxes is None:
        return None

    depth = camera_data.output["distance_to_image_plane"][0].squeeze(-1)
    depth = depth.cpu().numpy()
    height, width = depth.shape
    intrinsics = camera_data.intrinsic_matrices[0]
    polygons = result.masks.xy if result.masks is not None else None
    for index in result.boxes.conf.argsort(descending=True).tolist():
        if polygons is not None:
            mask = np.zeros(depth.shape, dtype=np.uint8)
            polygon = np.rint(polygons[index]).astype(np.int32)
            if len(polygon) < 3:
                continue
            cv2.fillPoly(mask, [polygon], 1)
            # Exclude edge pixels where depth can belong to the background.
            inner_mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
            region = inner_mask if inner_mask.any() else mask
        else:
            # Box-only weights can use the central 40% of the detection.
            x1, y1, x2, y2 = result.boxes.xyxy[index].tolist()
            left = max(0, int(x1 + 0.3 * (x2 - x1)))
            right = min(width, math.ceil(x2 - 0.3 * (x2 - x1)))
            top = max(0, int(y1 + 0.3 * (y2 - y1)))
            bottom = min(height, math.ceil(y2 - 0.3 * (y2 - y1)))
            region = np.zeros(depth.shape, dtype=bool)
            region[top:bottom, left:right] = True

        rows, cols = np.nonzero(region & np.isfinite(depth) & (depth > 0.1) & (depth < max_depth))
        if len(rows) < 3:
            continue
        values = depth[rows, cols]
        z = float(np.median(values))
        u = float(np.median(cols))
        v = float(np.median(rows))
        # ROS camera coordinates use +Z forward, +X right, and +Y down.
        point_camera = torch.stack(
            (
                (u - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (v - intrinsics[1, 2]) * z / intrinsics[1, 1],
                intrinsics.new_tensor(z),
            )
        )
        point_world = camera_data.pos_w[:1] + quat_apply(camera_data.quat_w_ros[:1], point_camera.unsqueeze(0))
        return point_world[0, :2]
    return None


def walk_to_detection(env, policy, model, target_class):
    """Acquire a stationary world target and let Stage 1 track it."""
    scene = env.unwrapped.scene
    camera = scene["head_camera"]
    robot = scene["robot"]
    command = env.unwrapped.command_manager.get_term("base_velocity")
    stop_distance = command.cfg.stop_distance
    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs

    samples = []
    goal = None
    settled_time = 0.0
    arrived = False
    step = 0
    last_detection_frame = -1
    print(f"[YOLO] Waiting for {args.target}.")
    with torch.inference_mode():
        while simulation_app.is_running():
            obs, _, dones, _ = env.step(policy(obs))
            step += 1
            if dones[0].item():
                goal = None
                samples.clear()
                settled_time = 0.0
                arrived = False
                last_detection_frame = -1
                print(f"[AISLE] Episode reset. Looking for {args.target} again.")
                continue

            if (goal is None or args.show) and step % args.detect_every == 0:
                data = camera.data
                frame_number = camera.frame[0].item()
                if goal is None and frame_number == last_detection_frame:
                    continue
                last_detection_frame = frame_number
                rgb = data.output["rgb"][0, ..., :3].cpu().numpy()
                frame = np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)
                result = model.predict(
                    frame, imgsz=max(frame.shape[:2]), classes=[target_class], conf=args.conf, verbose=False
                )[0]
                if args.show:
                    cv2.imshow("G1 head camera", result.plot())
                    cv2.waitKey(1)
                if goal is None:
                    point = detection_xy(data, result)
                    if point is None or not torch.isfinite(point).all().item():
                        samples.clear()
                        continue
                    samples.append(point)
                    if len(samples) < TARGET_SAMPLES:
                        continue
                    points = torch.stack(samples)
                    candidate = points.median(dim=0).values
                    if (points - candidate).norm(dim=1).amax().item() > TARGET_SPREAD_M:
                        samples.pop(0)
                        continue

                    goal = candidate
                    command.set_target_xy([0], goal)
                    # The next command update supplies Stage 1's existing target and yaw observations.
                    dx, dy = (goal - robot.data.root_pos_w[0, :2]).tolist()
                    yaw = command._root_yaw([0]).item()
                    bearing = math.remainder(math.atan2(dy, dx) - yaw, math.tau)
                    print(
                        f"[YOLO] Locked {args.target}: r={math.hypot(dx, dy):.2f} m, "
                        f"theta={math.degrees(bearing):+.1f} deg, world XY={goal.tolist()}"
                    )

            if goal is not None and not arrived:
                distance = command.target_distance[0].item()
                xy_speed = robot.data.root_lin_vel_w[0, :2].norm().item()
                yaw_error = abs(command.metrics["target_yaw_error"][0].item())
                yaw_rate = abs(robot.data.root_ang_vel_w[0, 2].item())
                settled = (
                    distance <= stop_distance
                    and xy_speed < 0.05
                    and yaw_error < math.radians(8)
                    and yaw_rate < 0.1
                )
                settled_time = settled_time + dt if settled else 0.0
                if settled_time >= SETTLE_SECONDS:
                    arrived = True
                    print(f"[AISLE] Settled {distance:.3f} m from the detected point.")
                    if args.target == "banana":
                        # Simulation diagnostic only. Navigation uses RGB-D exclusively.
                        banana_xy = goal.new_tensor(env.unwrapped.cfg.scene.target_object.init_state.pos[:2])
                        banana_xy += scene.env_origins[0, :2]
                        banana_distance = (banana_xy - robot.data.root_pos_w[0, :2]).norm().item()
                        print(f"[AISLE] Distance to the banana USD origin: {banana_distance:.3f} m.")


def main():
    args.target = args.target.strip().lower()
    model = YOLO(args.yolo_model)
    target_class = next((index for index, name in model.names.items() if name == args.target), None)
    if target_class is None:
        raise ValueError(f"YOLO weights {args.yolo_model!r} do not contain class {args.target!r}")

    env_cfg = parse_env_cfg(TASK, device=args.device, entry_point_key="play_env_cfg_entry_point")
    env_cfg.enable_aisle()
    agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, args)
    checkpoint = retrieve_file_path(args.checkpoint)
    print(f"[INFO] Loading Stage 1 checkpoint: {checkpoint}")
    env = gym.make(TASK, cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        walk_to_detection(env, policy, model, target_class)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if args.show:
            cv2.destroyAllWindows()
        simulation_app.close()
