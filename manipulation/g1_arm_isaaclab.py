"""Unitree G1 left-arm controller for integration into an existing Isaac Lab loop.

SIMULATION ONLY. This module does NOT step the simulator, alter other joints,
apply torques, or command hardware. The host supplies an Articulation, timestep,
and either a torso-frame wrist point or a world-frame grasp point (meters).

The returned joint-position targets must be merged with the host's existing
whole-body command. Host owns locomotion, actuator gains, collision monitoring,
and simulation stepping. NOT validated against a specific G1 USD asset.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import torch

ARM_NAMES = (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
)
# Original motor indices 15..21 are NOT Isaac Lab joint indices.
PRE_GRASP = torch.tensor([-0.25, 0.0, 0.0, 0.10, 0.0, -0.05, 0.0])
REACH_REFERENCE = torch.tensor([-0.75, 0.20, 0.0, 0.35, 0.0, 0.10, 0.0])


class Phase(str, Enum):
    IDLE = "IDLE"
    PRE_GRASP = "PRE_GRASP"
    REACH = "REACH"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    RETRACT = "RETRACT"
    RETURN_PRE_GRASP = "RETURN_PRE_GRASP"
    RETURN_HOME = "RETURN_HOME"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class ArmConfig:
    damping: float = 0.05
    max_ik_step_rad: float = 0.05
    max_speed_rad_s: float = 0.45
    reach_tolerance_m: float = 0.010
    joint_tolerance_rad: float = 0.09
    lift_m: float = 0.05
    retract_m: float = 0.10
    settle_s: float = 0.3
    stage_timeout_s: float = 12.0
    pregrasp_duration_s: float = 2.0
    return_duration_s: float = 3.0


def _rotation_inverse(q_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vector into local frame; q=[w,x,y,z]."""
    w = q_wxyz[..., :1]
    xyz = q_wxyz[..., 1:]
    t = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector - w * t + torch.cross(xyz, t, dim=-1)


class G1LeftArmController:
    """Single-environment (env_id) left-arm controller, no simulator ownership."""

    def __init__(self, robot, env_id: int = 0, config: Optional[ArmConfig] = None):
        self.robot = robot
        self.env_id = env_id
        self.cfg = config or ArmConfig()
        self.device = robot.data.joint_pos.device
        self.dtype = robot.data.joint_pos.dtype
        self.joint_ids = self._find_arm_joints()
        self.torso_id = self._find_body("torso_link")
        self.wrist_id = self._find_body("left_wrist_yaw_link")
        self.phase = Phase.IDLE
        self.elapsed = 0.0
        self.stable = 0.0
        self._last_command = self.arm_q().clone()
        self._phase_start = self._last_command.clone()
        self._home = self._last_command.clone()
        self._reach_xyz = None
        self._target_xyz = None
        self._failure_reason = None
        self._pickup_world = None
        self.tracking_error = None

    def _find_arm_joints(self):
        names = list(self.robot.joint_names)
        ids = []
        for short in ARM_NAMES:
            expected = f"left_{short}_joint"
            matches = [i for i, name in enumerate(names) if name == expected]
            if len(matches) != 1:
                raise ValueError(f"Expected one '{expected}' in Isaac Lab articulation; got {matches}. Available: {names}")
            ids.append(matches[0])
        return ids

    def _find_body(self, name):
        matches = [i for i, actual in enumerate(self.robot.body_names) if actual == name]
        if len(matches) != 1:
            raise ValueError(f"Body '{name}' missing/ambiguous. Available: {self.robot.body_names}")
        return matches[0]

    def arm_q(self):
        return self.robot.data.joint_pos[self.env_id, self.joint_ids].clone()

    def _limits(self):
        # Isaac Lab joint limits are (num_envs, num_joints, 2).
        lim = self.robot.data.soft_joint_pos_limits[self.env_id, self.joint_ids]
        return lim[:, 0], lim[:, 1]

    def _clip(self, q):
        lower, upper = self._limits()
        return torch.maximum(lower, torch.minimum(upper, q))

    def wrist_xyz_torso(self):
        d = self.robot.data
        torso_p = d.body_pos_w[self.env_id, self.torso_id]
        torso_q = d.body_quat_w[self.env_id, self.torso_id]
        wrist_p = d.body_pos_w[self.env_id, self.wrist_id]
        return _rotation_inverse(torso_q, wrist_p - torso_p)

    def _jacobian_torso(self):
        jac = self.robot.root_physx_view.get_jacobians()
        offset = jac.shape[-1] - self.robot.num_joints
        if offset not in (0, 6):
            raise RuntimeError(f"Unexpected PhysX Jacobian joint offset {offset}")
        body_offset = len(self.robot.body_names) - jac.shape[1]
        wrist_row = self.wrist_id - body_offset
        torso_row = self.torso_id - body_offset
        if wrist_row < 0 or wrist_row >= jac.shape[1]:
            raise RuntimeError("Wrist body has no valid PhysX Jacobian row")
        cols = [j + offset for j in self.joint_ids]
        jw = jac[self.env_id, wrist_row, :3, :][:, cols]
        if torso_row >= 0:
            jt = jac[self.env_id, torso_row, :3, :][:, cols]
            jw = jw - jt
        # Rotation of torso with respect to arm joints is normally zero.
        # If a custom articulation violates this, extend the relative Jacobian.
        torso_q = self.robot.data.body_quat_w[self.env_id, self.torso_id]
        return torch.stack([_rotation_inverse(torso_q, jw[:, i]) for i in range(7)], dim=1)

    def _ik_command(self, xyz_target, dt):
        error = xyz_target - self.wrist_xyz_torso()
        J = self._jacobian_torso()
        regularizer = (self.cfg.damping ** 2) * torch.eye(3, device=self.device, dtype=self.dtype)
        dq = J.T @ torch.linalg.solve(J @ J.T + regularizer, error)
        dq = dq.clamp(-self.cfg.max_ik_step_rad, self.cfg.max_ik_step_rad)
        dq = dq.clamp(-self.cfg.max_speed_rad_s * dt, self.cfg.max_speed_rad_s * dt)
        return self._clip(self.arm_q() + dq), float(torch.linalg.norm(error).item())

    def _smooth_command(self, goal, duration, dt):
        u = min(1.0, self.elapsed / max(duration, dt))
        alpha = 3 * u * u - 2 * u * u * u
        raw = self._phase_start + alpha * (goal - self._phase_start)
        # Rate limit relative to previous command (not current measured state).
        max_delta = self.cfg.max_speed_rad_s * dt
        return self._clip(self._last_command + (raw - self._last_command).clamp(-max_delta, max_delta))

    def _enter(self, phase):
        self.phase = phase
        self.elapsed = 0.0
        self.stable = 0.0
        self._phase_start = self.arm_q()

    def start(self, reach_xyz_torso):
        """Start from measured posture; target is WRIST XYZ in torso frame, meters.

        Caller must confirm reach_xyz_torso corresponds to the Isaac Lab asset,
        not copy MuJoCo world coordinates. No object pickup/gripper implemented.
        """
        if self.phase not in (Phase.IDLE, Phase.DONE, Phase.FAILED):
            raise RuntimeError("Controller already running")
        xyz = torch.as_tensor(reach_xyz_torso, dtype=self.dtype, device=self.device)
        if xyz.shape != (3,) or not bool(torch.isfinite(xyz).all()):
            raise ValueError("reach_xyz_torso must be 3 finite coordinates in meters")
        self._reach_xyz = xyz.clone()
        self._pickup_world = None
        self._home = self.arm_q()
        self._last_command = self._home.clone()
        self._failure_reason = None
        self._enter(Phase.PRE_GRASP)

    def reset(self):
        """Release pickup state after the host resets the environment."""
        self._pickup_world = None
        self._enter(Phase.IDLE)

    def start_pickup(self, target_xyz_w, facing_yaw):
        """Approach a shelf object with the Inspire grasp center in world coordinates."""
        from isaaclab.utils.math import quat_from_euler_xyz

        self.start([0.0, 0.0, 0.0])
        self._pickup_world = target_xyz_w.to(device=self.device, dtype=self.dtype).clone()
        if self._pickup_world.shape != (3,) or not bool(torch.isfinite(self._pickup_world).all()):
            self.reset()
            raise ValueError("Pickup requires a finite world XYZ point")
        yaw = self._pickup_world.new_tensor(facing_yaw)
        self._grasp_quat = quat_from_euler_xyz(yaw.new_tensor(torch.pi / 2), yaw.new_zeros(()), yaw)
        self._grasp_offset = self._pickup_world.new_tensor((0.185, -0.025, 0.0))
        self._approach_world = self._pickup_world + self._pickup_world.new_tensor((0.0, 0.0, 0.07))
        self._lift_world = self._pickup_world + self._pickup_world.new_tensor((0.0, 0.0, self.cfg.lift_m))
        self._phase_xyz_w, start_quat = self.grasp_pose_w()
        self._start_grasp_quat = torch.where((start_quat * self._grasp_quat).sum() < 0, -start_quat, start_quat)
        self.tracking_error = None

    def grasp_pose_w(self):
        """Position of the Inspire grasp center and orientation of the wrist link."""
        from isaaclab.utils.math import quat_apply

        data = self.robot.data
        pos = data.body_link_pos_w[self.env_id, self.wrist_id]
        quat = data.body_link_quat_w[self.env_id, self.wrist_id]
        return pos + quat_apply(quat.unsqueeze(0), self._grasp_offset.unsqueeze(0))[0], quat

    @property
    def finger_closure(self):
        if self.phase == Phase.CLOSE:
            return min(1.0, self.elapsed / 1.0)
        return float(self.phase in (Phase.LIFT, Phase.DONE))

    def _pickup_ik(self, xyz, goal_quat, dt):
        from isaaclab.utils.math import compute_pose_error

        point, quat = self.grasp_pose_w()
        pos_error, rot_error = compute_pose_error(
            point.unsqueeze(0), quat.unsqueeze(0), xyz.unsqueeze(0), goal_quat.unsqueeze(0),
            rot_error_type="axis_angle",
        )
        jacobians = self.robot.root_physx_view.get_jacobians()
        col_offset = jacobians.shape[-1] - self.robot.num_joints
        # Fixed-base PhysX Jacobians omit the root body.
        body_offset = len(self.robot.body_names) - jacobians.shape[1]
        row = self.wrist_id - body_offset
        if body_offset not in (0, 1) or col_offset not in (0, 6) or not 0 <= row < jacobians.shape[1]:
            raise RuntimeError(f"Unexpected wrist Jacobian layout {tuple(jacobians.shape)}")
        columns = [index + col_offset for index in self.joint_ids]
        jacobian = jacobians[self.env_id, row, :, :][:, columns]
        # PhysX returns a COM-referenced Jacobian. Shift it to the grasp center.
        lever = point - self.robot.data.body_com_pos_w[self.env_id, self.wrist_id]
        angular = jacobian[3:, :]
        linear = jacobian[:3, :] + torch.cross(angular.T, lever.expand(7, -1), dim=-1).T
        orientation_weight = 0.18
        task_jacobian = torch.cat((linear, orientation_weight * angular), dim=0)
        error = torch.cat((pos_error[0], orientation_weight * rot_error[0]))
        damping = (self.cfg.damping ** 2) * torch.eye(6, device=self.device, dtype=self.dtype)
        dq = task_jacobian.T @ torch.linalg.solve(task_jacobian @ task_jacobian.T + damping, error)
        desired = self._clip(self.arm_q() + dq.clamp(-0.15, 0.15))
        max_delta = self.cfg.max_speed_rad_s * dt
        cmd = self._clip(self._last_command + (desired - self._last_command).clamp(-max_delta, max_delta))
        return cmd, pos_error.norm().item(), rot_error.norm().item()

    def _step_pickup(self, dt):
        if self.phase in (Phase.DONE, Phase.FAILED):
            return self.joint_ids, self._last_command, self.phase
        self.elapsed += dt
        if self.elapsed > self.cfg.stage_timeout_s:
            return self.joint_ids, self._fail(f"Pickup timeout in {self.phase.value}"), self.phase
        goal, duration, next_phase = {
            Phase.PRE_GRASP: (self._approach_world, 3.0, Phase.REACH),
            Phase.REACH: (self._pickup_world, 1.5, Phase.CLOSE),
            Phase.CLOSE: (self._pickup_world, 1.25, Phase.LIFT),
            Phase.LIFT: (self._lift_world, 2.0, Phase.DONE),
        }[self.phase]
        u = min(1.0, self.elapsed / duration)
        blend = u * u * (3.0 - 2.0 * u)
        xyz = torch.lerp(self._phase_xyz_w, goal, blend)
        goal_quat = self._grasp_quat
        if self.phase == Phase.PRE_GRASP:
            goal_quat = torch.lerp(self._start_grasp_quat, goal_quat, blend)
            goal_quat = goal_quat / goal_quat.norm()
        cmd, position_error, rotation_error = self._pickup_ik(xyz, goal_quat, dt)
        self.tracking_error = (position_error, rotation_error)
        if not bool(torch.isfinite(cmd).all()):
            return self.joint_ids, self._fail("Non-finite pickup IK result"), self.phase
        self._last_command = cmd
        close_enough = position_error < self.cfg.reach_tolerance_m and rotation_error < 0.35
        if self.phase == Phase.CLOSE:
            # Torque-limited fingers can displace the wrist while contacting the fruit.
            close_enough = position_error < 0.04 and rotation_error < 0.45
        self.stable = self.stable + dt if u == 1.0 and close_enough else 0.0
        if self.stable >= self.cfg.settle_s:
            self._phase_xyz_w, _ = self.grasp_pose_w()
            self._enter(next_phase)
        return self.joint_ids, cmd, self.phase

    @property
    def failure_reason(self):
        return self._failure_reason

    def _fail(self, message):
        self._failure_reason = message
        self.phase = Phase.FAILED
        # Freeze output at measured posture; host should enter its safe mode.
        self._last_command = self.arm_q()
        return self._last_command.clone()

    def step(self, dt: float):
        """Call ONCE per physics/control tick after state refresh.

        Returns (joint_ids, q_target, phase). Never writes to the robot.
        Host must retain its own non-arm targets and merge only these 7 joints.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if self._pickup_world is not None:
            return self._step_pickup(dt)
        if self.phase in (Phase.IDLE, Phase.DONE, Phase.FAILED):
            return self.joint_ids, self.arm_q(), self.phase
        self.elapsed += dt
        if self.elapsed > self.cfg.stage_timeout_s:
            return self.joint_ids, self._fail(f"Timeout in {self.phase.value}"), self.phase

        if self.phase in (Phase.PRE_GRASP, Phase.RETURN_PRE_GRASP, Phase.RETURN_HOME):
            if self.phase == Phase.RETURN_HOME:
                goal = self._home
            else:
                goal = PRE_GRASP.to(device=self.device, dtype=self.dtype)
            duration = (self.cfg.pregrasp_duration_s if self.phase == Phase.PRE_GRASP
                        else self.cfg.return_duration_s)
            cmd = self._smooth_command(goal, duration, dt)
            within = float((self.arm_q() - goal).abs().max()) < self.cfg.joint_tolerance_rad
            within = within and self.elapsed >= duration
        else:
            if self.phase == Phase.REACH:
                xyz = self._reach_xyz
            elif self.phase == Phase.LIFT:
                xyz = self._reach_xyz + torch.tensor([0., 0., self.cfg.lift_m], device=self.device, dtype=self.dtype)
            else:
                xyz = self._reach_xyz + torch.tensor([-self.cfg.retract_m, 0., self.cfg.lift_m], device=self.device, dtype=self.dtype)
            cmd, error_m = self._ik_command(xyz, dt)
            within = error_m < self.cfg.reach_tolerance_m

        self._last_command = cmd.clone()
        self.stable = self.stable + dt if within else 0.0
        if self.stable >= self.cfg.settle_s:
            next_phase = {
                Phase.PRE_GRASP: Phase.REACH,
                Phase.REACH: Phase.LIFT,
                Phase.LIFT: Phase.RETRACT,
                Phase.RETRACT: Phase.RETURN_PRE_GRASP,
                Phase.RETURN_PRE_GRASP: Phase.RETURN_HOME,
                Phase.RETURN_HOME: Phase.DONE,
            }[self.phase]
            self._enter(next_phase)
        return self.joint_ids, cmd, self.phase
