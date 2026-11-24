"""Walk the Stage 1 G1 policy to the banana, then reach, grasp, and lift it.

Run from unitree_rl_lab with --checkpoint /path/to/stage1.pt.
Use --show to display detections in a desktop session.
"""

import argparse
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target", default="banana", help="YOLO class to approach.")
parser.add_argument("--yolo_model", default="yolov8n-seg.pt", help="YOLO weights file.")
parser.add_argument("--conf", type=float, default=0.25, help="Minimum YOLO confidence.")
parser.add_argument("--detect_every", type=int, default=5, help="Steps between detections.")
parser.add_argument("--show", action="store_true", help="Show the head-camera detections.")
parser.add_argument("--inspire_urdf", help="Path to the G1 Inspire DFQ URDF in unitree_ros.")
parser.add_argument("--walk_only", action="store_true", help="Run the aisle demo with the original walking robot.")
parser.add_argument("--ik_diagnostic", action="store_true", help="Test one fixed-base grasp using the simulated banana pose.")
parser.add_argument("--record_pickup_trace", type=Path, help="Save a successful diagnostic's arm and finger commands for training.")
parser.add_argument("--pickup_debug", action="store_true", help="Print tracking errors and left-arm contacts during pickup.")
parser.add_argument("--pickup_standoff", type=float, default=0.45, help="Across-aisle approach offset in metres.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint and not args.ik_diagnostic:
    parser.error("--checkpoint must point to a trained walking or pickup-balance policy")
if args.ik_diagnostic and (args.walk_only or args.show or args.target.strip().lower() != "banana"):
    parser.error("--ik_diagnostic tests the banana directly and cannot be combined with --walk_only or --show")
if args.record_pickup_trace and not args.ik_diagnostic:
    parser.error("--record_pickup_trace requires --ik_diagnostic")
if not math.isfinite(args.pickup_standoff) or args.pickup_standoff <= 0:
    parser.error("--pickup_standoff must be positive and finite")
if args.detect_every < 1:
    parser.error("--detect_every must be at least 1")
args.enable_cameras = not args.ik_diagnostic
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
from unitree_rl_lab.assets.robots.unitree import INSPIRE_LEFT_GRASP
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

TASK = "Unitree-G1-29dof-Velocity"
TARGET_SAMPLES = 2  # lock the target on the first detection with valid depth
TARGET_SPREAD_M = 0.15
SETTLE_SECONDS = 0.5
# Stop in the aisle in front of the object instead of walking straight at it (it sits inside the shelf).
APPROACH_STOP_M = 0.20  # how close to the approach spot counts as "there"


def detection_xyz(camera_data, result, max_depth=6.0):
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
        return point_world[0]
    return None


class BananaPickup:
    """Coordinate the left arm and Inspire fingers while PPO controls the remaining joints."""

    def __init__(self, env):
        # The manipulation controller lives alongside unitree_rl_lab in the SHOP checkout.
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        try:
            from manipulation.g1_arm_isaaclab import ArmConfig, G1LeftArmController, Phase
        finally:
            sys.path.pop(0)
        self.Phase = Phase
        self.robot = env.scene["robot"]
        self.banana = env.scene["target_object"]
        self.action = env.action_manager.get_term("JointPositionAction")
        self.command = env.command_manager.get_term("base_velocity")
        self.contact_sensor = env.scene["contact_forces"]
        self.arm_contact_ids, _ = self.contact_sensor.find_bodies(
            r"left_(shoulder|elbow|wrist).*|L_.*"
        )
        self.arm = G1LeftArmController(
            self.robot, config=ArmConfig(max_speed_rad_s=0.40, reach_tolerance_m=0.022, lift_m=0.09)
        )
        self.finger_ids = [self.robot.joint_names.index(name) for name in INSPIRE_LEFT_GRASP]
        self.closed = self.robot.data.joint_pos.new_tensor(list(INSPIRE_LEFT_GRASP.values())).unsqueeze(0)
        self.reset()

    def reset(self):
        self.arm.reset()
        self.start_height = None
        self.lift_hold = 0.0
        self.done_time = 0.0
        self.result_printed = False
        self.closure = 0.0
        self.debug_elapsed = 0.0
        self.peak_drift = self.closed.new_zeros(())

    def start(self, goal, heading):
        self.arm.start_pickup(goal, heading)
        self.action.begin_pickup([0])
        self.start_height = self.banana.data.root_pos_w[0, 2].item()
        print(f"[PICKUP] Reaching toward detected XYZ {goal.tolist()}.")

    def step(self, dt):
        if self.start_height is None:
            self.robot.set_joint_position_target(self.closed * 0.0, joint_ids=self.finger_ids)
            return

        phase_before = self.arm.phase
        _, q_target, phase = self.arm.step(dt)
        if phase != self.Phase.FAILED:
            self.closure = self.arm.finger_closure
        self.action.set_arm_targets([0], q_target)
        self.robot.set_joint_position_target(self.closed * self.closure, joint_ids=self.finger_ids)
        drift = (self.robot.data.root_pos_w[0, :2] - self.command.hold_xy_w[0]).norm()
        self.peak_drift = torch.maximum(self.peak_drift, drift)
        if phase != phase_before:
            print(f"[PICKUP] {phase.value}")
        if (args.pickup_debug or args.ik_diagnostic) and not self.result_printed:
            self.debug_elapsed += dt
            if self.debug_elapsed >= 0.5 or phase != phase_before:
                self.report_tracking(phase)
                self.debug_elapsed = 0.0
        if self.result_printed:
            return q_target
        if phase == self.Phase.FAILED:
            self.result_printed = True
            print(f"[PICKUP] {self.arm.failure_reason}. Peak base drift: {self.peak_drift.item():.3f} m.")
        elif phase == self.Phase.DONE:
            self.done_time += dt
            fruit = self.banana.data.root_pos_w[0]
            center, _ = self.arm.grasp_pose_w()
            lift = fruit[2].item() - self.start_height
            holding = lift >= 0.045 and (fruit - center).norm().item() < 0.13
            self.lift_hold = self.lift_hold + dt if holding else 0.0
            if self.lift_hold >= 0.4 or self.done_time >= 2.0:
                status = "Picked up banana" if self.lift_hold >= 0.4 else "Grasp not verified"
                print(f"[PICKUP] {status}. Banana lift: {lift:.3f} m. Peak base drift: {self.peak_drift.item():.3f} m.")
                self.result_printed = True
        return q_target

    def report_tracking(self, phase):
        if self.arm.tracking_error is None:
            return
        pos, rot = self.arm.tracking_error
        lower, upper = self.robot.data.soft_joint_pos_limits[0, self.arm.joint_ids].unbind(dim=-1)
        q = self.arm.arm_q()
        near_limit = ((q - lower < 0.03) | (upper - q < 0.03)).sum().item()
        contact_peak = 0.0
        if self.arm_contact_ids:
            forces = self.contact_sensor.data.net_forces_w_history[0, :, self.arm_contact_ids]
            contact_peak = forces.norm(dim=-1).amax().item()
        yaw_error = math.remainder((self.command.hold_yaw_w[0] - self.command._root_yaw([0])[0]).item(), math.tau)
        print(
            f"[PICKUP DEBUG] {phase.value} pos={pos:.3f} m rot={math.degrees(rot):.1f} deg "
            f"joints_near_limit={near_limit} arm_contact={contact_peak:.1f} N "
            f"peak_drift={self.peak_drift.item():.3f} m yaw_error={math.degrees(yaw_error):+.1f} deg"
        )


def walk_to_detection(env, policy, model, target_class, pickup=None):
    """Acquire a stationary world target and let Stage 1 track it."""
    scene = env.unwrapped.scene
    camera = scene["head_camera"]
    robot = scene["robot"]
    command = env.unwrapped.command_manager.get_term("base_velocity")
    original_stop = command.cfg.stop_distance
    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs

    samples = []
    goal = None
    nearby_xyz = None
    settled_time = 0.0
    arrived = False
    facing = False  # False = walking to the approach spot, True = turning to face the object
    approach_time = 0.0
    step = 0
    last_detection_frame = -1
    print(f"[YOLO] Waiting for {args.target}.")
    with torch.inference_mode():
        while simulation_app.is_running():
            if pickup is not None:
                pickup.step(dt)
            obs, _, dones, _ = env.step(policy(obs))
            step += 1
            if dones[0].item():
                goal = None
                nearby_xyz = None
                samples.clear()
                settled_time = 0.0
                arrived = False
                facing = False
                approach_time = 0.0
                command.cfg.stop_distance = original_stop
                last_detection_frame = -1
                if pickup is not None:
                    pickup.reset()
                print(f"[AISLE] Episode reset. Looking for {args.target} again.")
                continue

            close_to_goal = goal is not None and command.target_distance[0].item() < 0.65
            refine_grasp = pickup is not None and close_to_goal and not arrived
            if (goal is None or refine_grasp or args.show) and step % args.detect_every == 0:
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
                    point = detection_xyz(data, result)
                    if point is None or not torch.isfinite(point).all().item():
                        samples.clear()
                        continue
                    samples.append(point)
                    if len(samples) < TARGET_SAMPLES:
                        continue
                    points = torch.stack(samples)
                    candidate = points.median(dim=0).values
                    if (points[:, :2] - candidate[:2]).norm(dim=1).amax().item() > TARGET_SPREAD_M:
                        samples.pop(0)
                        continue

                    goal = candidate
                    # Walk to a spot in the aisle in front of the object (on the aisle side of it).
                    side = 1.0 if (goal[1] - scene.env_origins[0, 1]).item() >= 0.0 else -1.0
                    approach = goal[:2].clone()
                    approach[1] -= side * args.pickup_standoff
                    command.cfg.stop_distance = APPROACH_STOP_M
                    command.set_target_xy([0], approach)
                    print(f"[AISLE] Walking to the spot in front of it: XY={approach.tolist()}")
                    # The next command update supplies Stage 1's existing target and yaw observations.
                    dx, dy = (goal[:2] - robot.data.root_pos_w[0, :2]).tolist()
                    yaw = command._root_yaw([0]).item()
                    bearing = math.remainder(math.atan2(dy, dx) - yaw, math.tau)
                    print(
                        f"[YOLO] Locked {args.target}: r={math.hypot(dx, dy):.2f} m, "
                        f"theta={math.degrees(bearing):+.1f} deg, world XYZ={goal.tolist()}"
                    )
                elif refine_grasp:
                    point = detection_xyz(data, result)
                    if point is not None and (point - goal).norm().item() < 0.12:
                        nearby_xyz = point if nearby_xyz is None else torch.lerp(nearby_xyz, point, 0.35)

            if goal is not None and not arrived and not facing:
                # Step 1: walk to the approach spot. Once stopped there, aim at the object itself,
                # with a stop distance larger than the standoff so it only turns on the spot.
                distance = command.target_distance[0].item()
                xy_speed = robot.data.root_lin_vel_w[0, :2].norm().item()
                at_spot = distance <= command.cfg.stop_distance and xy_speed < 0.08
                approach_time = approach_time + dt if at_spot else 0.0
                if approach_time >= SETTLE_SECONDS:
                    facing = True
                    command.cfg.stop_distance = original_stop
                    command.set_target_xy([0], goal[:2])
                    print("[AISLE] In front of the shelf. Turning to face the object.")

            elif goal is not None and not arrived:
                # Step 2: turning to face the object, then settle and start the pickup.
                distance = command.target_distance[0].item()
                xy_speed = robot.data.root_lin_vel_w[0, :2].norm().item()
                yaw_error = abs(command.metrics["target_yaw_error"][0].item())
                yaw_rate = abs(robot.data.root_ang_vel_w[0, 2].item())
                if step % 50 == 0:
                    print(
                        f"[DEBUG] dist={distance:.2f} (need <= {command.cfg.stop_distance:.2f})  "
                        f"speed={xy_speed:.3f} (need < 0.05)  "
                        f"yaw_err={math.degrees(yaw_error):.1f} deg (need < 15)  "
                        f"yaw_rate={yaw_rate:.2f} (need < 0.3)"
                    )
                settled = (
                    distance <= command.cfg.stop_distance
                    and xy_speed < 0.05
                    and yaw_error < math.radians(15)
                    and yaw_rate < 0.3
                )
                settled_time = settled_time + dt if settled else 0.0
                if settled_time >= SETTLE_SECONDS:
                    arrived = True
                    print(f"[AISLE] Settled {distance:.3f} m from the detected point.")
                    if pickup is not None:
                        target = goal if nearby_xyz is None else nearby_xyz
                        dx, dy = (target[:2] - robot.data.root_pos_w[0, :2]).tolist()
                        heading = math.atan2(dy, dx)  # point the grasp at the banana even if the body is a few degrees off
                        pickup.start(target, heading)
                    if args.target == "banana":
                        # Simulation diagnostic only. Navigation uses RGB-D exclusively.
                        banana_xy = goal.new_tensor(env.unwrapped.cfg.scene.target_object.init_state.pos[:2])
                        banana_xy += scene.env_origins[0, :2]
                        banana_distance = (banana_xy - robot.data.root_pos_w[0, :2]).norm().item()
                        print(f"[AISLE] Distance to the banana USD origin: {banana_distance:.3f} m.")


@torch.inference_mode()
def run_ik_diagnostic(env, heading):
    env.reset()
    pickup = BananaPickup(env)
    if not pickup.robot.is_fixed_base:
        raise RuntimeError("IK diagnostic requires a fixed-base Inspire articulation")
    zeros = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    dt = env.step_dt
    for _ in range(math.ceil(SETTLE_SECONDS / dt)):
        env.step(zeros)
    pickup.start(pickup.banana.data.root_pos_w[0], heading)
    samples = []
    while simulation_app.is_running():
        q_target = pickup.step(dt)
        if args.record_pickup_trace:
            samples.append(torch.cat((q_target, q_target.new_tensor([pickup.closure]))))
        if pickup.result_printed:
            if pickup.lift_hold < 0.4:
                raise RuntimeError("Fixed-base pickup failed. Check the phase and tracking log above.")
            if args.record_pickup_trace:
                with args.record_pickup_trace.open("x", encoding="utf-8") as file:
                    json.dump({"dt": dt, "samples": torch.stack(samples).cpu().tolist()}, file)
                print(f"[PICKUP] Saved training trace to {args.record_pickup_trace}")
            return
        _, _, terminated, timed_out, _ = env.step(zeros)
        if terminated[0].item() or timed_out[0].item():
            raise RuntimeError("Fixed-base diagnostic reset before pickup completed")


def main():
    args.target = args.target.strip().lower()
    if not args.walk_only and args.target != "banana":
        raise ValueError("Pickup is configured for banana. Use --walk_only to approach another class.")
    env_cfg = parse_env_cfg(TASK, device=args.device, entry_point_key="play_env_cfg_entry_point")
    checkpoint = retrieve_file_path(args.checkpoint) if args.checkpoint else None
    env_cfg.enable_aisle(pickup=not args.walk_only)
    if not args.walk_only:
        env_cfg.enable_inspire(args.inspire_urdf, checkpoint)
    if args.ik_diagnostic:
        env_cfg.scene.head_camera = None
        robot_cfg = env_cfg.scene.robot
        robot_cfg.spawn.fix_base = True
        x, y, _ = env_cfg.scene.target_object.init_state.pos
        heading = math.copysign(math.pi / 2, y)
        robot_cfg.init_state.pos = (x, y - math.copysign(args.pickup_standoff, y), robot_cfg.init_state.pos[2])
        robot_cfg.init_state.rot = (math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2))
        env = gym.make(TASK, cfg=env_cfg)
        try:
            run_ik_diagnostic(env.unwrapped, heading)
        finally:
            env.close()
        return

    model = YOLO(args.yolo_model)
    target_class = next((index for index, name in model.names.items() if name == args.target), None)
    if target_class is None:
        raise ValueError(f"YOLO weights {args.yolo_model!r} do not contain class {args.target!r}")

    agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, args)
    print(f"[INFO] Loading policy checkpoint: {checkpoint}")
    env = gym.make(TASK, cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        pickup = None if args.walk_only else BananaPickup(env.unwrapped)
        walk_to_detection(env, policy, model, target_class, pickup)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if args.show:
            cv2.destroyAllWindows()
        simulation_app.close()
