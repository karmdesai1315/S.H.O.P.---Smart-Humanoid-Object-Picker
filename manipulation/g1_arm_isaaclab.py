"""Unitree G1 left-arm controller for integration into an existing Isaac Lab loop.

SIMULATION ONLY. This module does NOT step the simulator, alter other joints,
apply torques, or command hardware. The host supplies an Articulation, timestep,
and a target wrist XYZ expressed in torso_link coordinates (meters).

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
        # PhysX view: fixed base (N, B, 6, J), floating base
        # (N, B-1, 6, J+6). For floating base, root body is omitted.
        jac = self.robot.root_physx_view.get_jacobians()
        offset = jac.shape[-1] - self.robot.num_joints
        if offset not in (0, 6):
            raise RuntimeError(f"Unexpected PhysX Jacobian joint offset {offset}")
        body_offset = 1 if offset == 6 else 0
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
        self._home = self.arm_q()
        self._last_command = self._home.clone()
        self._failure_reason = None
        self._enter(Phase.PRE_GRASP)

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
