"""S.H.O.P. G1 dual-mode left-arm manipulation with 6D damped least-squares IK.

ISAAC LAB 2.x / PyTorch; simulation integration module, NOT a standalone app.
No DDS/MuJoCo, no actuator/gripper closure, no self-collision solver, no balancing.
Host MUST provide an Isaac-asset collision/scene safety validator and merge the
seven arm position targets with its existing whole-body/balance commands.

Task frame: torso_link. Joint order: shoulder pitch/roll/yaw, elbow,
wrist roll/pitch/yaw. Wrist point: left_wrist_yaw_link origin (not a grasp TCP).
Quaternion format: (w, x, y, z), as in Isaac Lab 2.x body_quat_w.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence
import math
import torch

ARM = ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow',
       'wrist_roll','wrist_pitch','wrist_yaw')
JOINT_NAMES = tuple(f'left_{n}_joint' for n in ARM)
WRIST_ROLL = 4


class Mode(str, Enum):
    NORMAL = 'normal'
    FALLEN_OBJECT = 'fallen_object'


class Phase(str, Enum):
    IDLE='IDLE'
    PRE_GRASP='PRE_GRASP'
    REACH='REACH'
    ROLL_1='ROLL_1'
    ROLL_2='ROLL_2'
    PALM_DOWN='PALM_DOWN'
    DESCEND='DESCEND'
    LIFT='LIFT'
    RETRACT='RETRACT'
    RETURN_PRE_GRASP='RETURN_PRE_GRASP'
    RETURN_HOME='RETURN_HOME'
    DONE='DONE'
    FAILED='FAILED'


@dataclass
class ArmConfig:
    pre_grasp_rad: tuple[float,...] = (-.25,0.,0.,.10,0.,-.05,0.)
    palm_roll_rad: float = math.pi/2  # empirically tuned in MuJoCo at PRE_GRASP ONLY
    descent_m: float = .05
    lift_m: float = .05
    retract_m: float = .10
    damping: float = .06
    orientation_weight_m_per_rad: float = .15
    max_speed_rad_s: float = .75
    max_increment_rad: float = .05
    nullspace_gain: float = .05
    wrist_pitch_preference_rad: float = 0.
    reach_tolerance_m: float = .010
    orientation_tolerance_rad: float = math.radians(5)
    stage_extra_s: float = 5.
    # Preserve original normal-mode position-only manipulation behavior.
    normal_hold_orientation: bool = False


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton multiplication of (w,x,y,z) unit quaternions."""
    w1,x1,y1,z1 = a.unbind(-1)
    w2,x2,y2,z2 = b.unbind(-1)
    return torch.stack((w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2),dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1],-q[..., 1:]),dim=-1)


def qnormalize(q: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(q).all()) or float(torch.linalg.vector_norm(q)) < .9:
        raise ValueError('Invalid Isaac body quaternion')
    return q / torch.linalg.vector_norm(q)


def quat_rotate(q:torch.Tensor,v:torch.Tensor)->torch.Tensor:
    q=qnormalize(q)
    t=2*torch.cross(q[1:],v,dim=0)
    return v + q[0]*t + torch.cross(q[1:],t,dim=0)


def quat_inv_rotate(q:torch.Tensor,v:torch.Tensor)->torch.Tensor:
    return quat_rotate(quat_conj(qnormalize(q)),v)


def quat_error_axis_angle(desired:torch.Tensor,current:torch.Tensor)->torch.Tensor:
    """Rotation vector taking current frame orientation to desired (same basis)."""
    e=qnormalize(quat_mul(qnormalize(desired),quat_conj(qnormalize(current))))
    if float(e[0]) < 0.: e=-e  # shortest path
    v=e[1:]
    n=torch.linalg.vector_norm(v)
    if float(n) < 1e-7: return 2*v
    angle=2*torch.atan2(n,e[0])
    return angle*v/n


class G1Arm6D:
    """Single-environment target generator. Host owns sim stepping and control.

    `is_segment_safe(measured_arm_q, next_arm_cmd, phase)` is REQUIRED.
    It must verify limits, collisions, clearance and unknown/obstacle space in
    the team's real Isaac USD scene, and return False if uncertain. This module
    does NOT provide a fake safe validator. No guarantee of real-robot safety.
    """
    def __init__(self,robot,*,env_id:int=0,is_segment_safe:Optional[Callable]=None,
                 config:Optional[ArmConfig]=None):
        self.robot=robot; self.env_id=int(env_id); self.cfg=config or ArmConfig()
        data=robot.data
        self.device=data.joint_pos.device; self.dtype=data.joint_pos.dtype
        if not 0 <= self.env_id < data.joint_pos.shape[0]:
            raise ValueError('Invalid env_id')
        self.joint_ids=[self._unique_id(robot.joint_names,n) for n in JOINT_NAMES]
        self.torso_id=self._unique_id(robot.body_names,'torso_link')
        self.wrist_id=self._unique_id(robot.body_names,'left_wrist_yaw_link')
        self.is_segment_safe=is_segment_safe
        self.phase=Phase.IDLE; self.mode=None; self.reason=None
        self.reach_error_m=None; self.orientation_error_rad=None
        self.elapsed=0.; self._stage_index=-1; self._sequence=[]
        self._last_cmd=self.arm_q(); self._home=self._last_cmd.clone()
        self._stage_start=self._last_cmd.clone(); self._goal=None
        self._reach_xyz=None; self._reach_quat=None
        self._palm_xyz=None; self._palm_quat=None
        self._roll_initial=0.

    @staticmethod
    def _unique_id(names,name):
        matches=[i for i,n in enumerate(names) if n==name]
        if len(matches)!=1: raise ValueError(f'Missing/ambiguous Isaac name: {name}')
        return matches[0]

    def arm_q(self):
        return self.robot.data.joint_pos[self.env_id,self.joint_ids].detach().clone()

    def _limits(self):
        a=self.robot.data.soft_joint_pos_limits[self.env_id,self.joint_ids]
        return a[:,0],a[:,1]

    def _check_q(self,q):
        lo,hi=self._limits()
        if q.shape!=(7,) or not bool(torch.isfinite(q).all()) or not bool(((q>=lo)&(q<=hi)).all()):
            raise ValueError('Nonfinite / outside Isaac soft joint limits')
        return q

    def wrist_pose_torso(self):
        d=self.robot.data; e=self.env_id
        torso_p=d.body_pos_w[e,self.torso_id]; wrist_p=d.body_pos_w[e,self.wrist_id]
        torso_q=qnormalize(d.body_quat_w[e,self.torso_id])
        wrist_q=qnormalize(d.body_quat_w[e,self.wrist_id])
        return (quat_inv_rotate(torso_q,wrist_p-torso_p),
                qnormalize(quat_mul(quat_conj(torso_q),wrist_q)))

    def jacobian_6d_torso(self):
        """6x7 of torso-relative wrist link pose, using world spatial jacobians.

        Account for floating-base +6 columns and fixed-base omitted root row.
        For p_rel=R_t^T(p_w-p_t):
          Jp_rel=R_t^T[Jp_w-Jp_t + (p_w-p_t) x Jomega_t]
          Jomega_rel=R_t^T[Jomega_w-Jomega_t]
        The cross term is zero for arm joints if torso motion is independent.
        NOTE: raw PhysX Jacobian linear rows may be COM-referenced on some
        Isaac Lab releases/assets. Verify wrist link-origin FK vs FD Jacobian.
        """
        raw=self.robot.root_physx_view.get_jacobians()
        if raw.ndim!=4 or raw.shape[-2]!=6:
            raise RuntimeError('Unexpected PhysX jacobian array shape')
        offset=raw.shape[-1]-self.robot.num_joints
        if offset not in (0,6):
            raise RuntimeError(f'Unexpected PhysX jacobian base columns: {offset}')
        row_offset=1 if offset==0 else 0
        wrist_row=self.wrist_id-row_offset; torso_row=self.torso_id-row_offset
        if not (0<=wrist_row<raw.shape[1]):
            raise RuntimeError('Wrist Jacobian body row unavailable')
        columns=[i+offset for i in self.joint_ids]
        Jw=raw[self.env_id,wrist_row,:,:][:,columns].clone()
        Jt=(raw[self.env_id,torso_row,:,:][:,columns].clone()
            if 0<=torso_row<raw.shape[1] else torch.zeros_like(Jw))
        d=self.robot.data
        q_t=qnormalize(d.body_quat_w[self.env_id,self.torso_id]);
        dp=d.body_pos_w[self.env_id,self.wrist_id]-d.body_pos_w[self.env_id,self.torso_id]
        linear_world=Jw[:3]-Jt[:3]+torch.cross(dp.expand(7,3),Jt[3:].T,dim=1).T
        angular_world=Jw[3:]-Jt[3:]
        linear=torch.stack([quat_inv_rotate(q_t,linear_world[:,i]) for i in range(7)],dim=1)
        angular=torch.stack([quat_inv_rotate(q_t,angular_world[:,i]) for i in range(7)],dim=1)
        J=torch.cat((linear,angular),dim=0)
        if not bool(torch.isfinite(J).all()): raise ValueError('Invalid Jacobian')
        return J

    def _safety(self,a,b,phase):
        if self.is_segment_safe is None:
            raise RuntimeError('Missing required Isaac USD collision validator')
        return bool(self.is_segment_safe(a.detach().clone(),b.detach().clone(),phase))

    def start(self,mode:Mode|str,reach_xyz_torso:Sequence[float],
              reach_quat_torso:Optional[Sequence[float]]=None):
        """Start with torso-relative wrist position (NOT raw object center).

        Optional REACH quaternion (w,x,y,z) is used as 6D REACH goal.
        Without it, only XYZ is controlled at REACH; for later 6D phases
        the measured wrist orientation after the roll becomes the reference.
        """
        if self.phase not in (Phase.IDLE,Phase.DONE,Phase.FAILED):
            raise RuntimeError('Manipulator already active')
        mode=Mode(mode)
        xyz=torch.as_tensor(reach_xyz_torso,dtype=self.dtype,device=self.device)
        if xyz.shape!=(3,) or not bool(torch.isfinite(xyz).all()):
            raise ValueError('Need finite torso-frame wrist XYZ')
        quat=(None if reach_quat_torso is None else qnormalize(torch.as_tensor(
            reach_quat_torso,dtype=self.dtype,device=self.device)))
        pre=torch.tensor(self.cfg.pre_grasp_rad,dtype=self.dtype,device=self.device)
        self._check_q(pre)
        if mode==Mode.FALLEN_OBJECT:
            low,high=self._limits()
            if not float(low[WRIST_ROLL])<=self.cfg.palm_roll_rad<=float(high[WRIST_ROLL]):
                raise ValueError('Calibrated wrist roll outside Isaac soft limits')
        if not self._safety(self.arm_q(),pre,Phase.PRE_GRASP):
            raise RuntimeError('Initial PRE_GRASP path unsafe or unknown')
        self.mode=mode; self.reason=None; self.reach_error_m=None
        self.orientation_error_rad=None
        self._reach_xyz=xyz.detach().clone(); self._reach_quat=quat
        self._home=self.arm_q(); self._last_cmd=self._home.clone()
        self._palm_xyz=None; self._palm_quat=None
        self._sequence=([Phase.PRE_GRASP,Phase.REACH,Phase.LIFT,Phase.RETRACT,
                         Phase.RETURN_PRE_GRASP,Phase.RETURN_HOME]
                         if mode==Mode.NORMAL else
                        [Phase.PRE_GRASP,Phase.REACH,Phase.ROLL_1,Phase.ROLL_2,
                         Phase.PALM_DOWN,Phase.DESCEND,Phase.LIFT,Phase.RETRACT,
                         Phase.RETURN_PRE_GRASP,Phase.RETURN_HOME])
        self._stage_index=-1; self._advance()

    def _advance(self):
        self._stage_index+=1; self.elapsed=0.; self._goal=None
        self._stage_start=self._last_cmd.clone()  # NEVER rebase to measured q
        if self._stage_index>=len(self._sequence):
            self.phase=Phase.DONE; return
        self.phase=self._sequence[self._stage_index]
        if self.phase in (Phase.PRE_GRASP,Phase.RETURN_PRE_GRASP):
            self._goal=torch.tensor(self.cfg.pre_grasp_rad,dtype=self.dtype,device=self.device)
        elif self.phase==Phase.RETURN_HOME: self._goal=self._home.clone()
        elif self.phase==Phase.ROLL_1:
            self._roll_initial=float(self._last_cmd[WRIST_ROLL])
        if self.phase in (Phase.ROLL_1,Phase.ROLL_2,Phase.PALM_DOWN):
            fraction={Phase.ROLL_1:1/3,Phase.ROLL_2:2/3,Phase.PALM_DOWN:1.}[self.phase]
            self._goal=self._stage_start.clone()
            self._goal[WRIST_ROLL]=(self._roll_initial+fraction*(
                self.cfg.palm_roll_rad-self._roll_initial))
        if self.phase==Phase.DESCEND:
            # Capture actual palm pose when roll phase finishes: honest about
            # tracking lag. Maintain this orientation through 6D IK phases.
            self._palm_xyz,self._palm_quat=self.wrist_pose_torso()
            self._palm_xyz=self._palm_xyz.detach().clone()
            self._palm_quat=self._palm_quat.detach().clone()
        if self.phase==Phase.LIFT and self.mode==Mode.NORMAL and self.cfg.normal_hold_orientation:
            _,self._palm_quat=self.wrist_pose_torso()
            self._palm_quat=self._palm_quat.detach().clone()
        if self._goal is not None: self._check_q(self._goal)

    def _duration(self):
        return {Phase.PRE_GRASP:2.,Phase.REACH:3.,Phase.ROLL_1:1.5,
            Phase.ROLL_2:1.5,Phase.PALM_DOWN:1.5,Phase.DESCEND:3.,
            Phase.LIFT:3.,Phase.RETRACT:3.,Phase.RETURN_PRE_GRASP:3.,
            Phase.RETURN_HOME:3.}[self.phase]

    def _cartesian_target(self):
        d=self.cfg
        if self.phase==Phase.REACH: return self._reach_xyz,self._reach_quat
        if self.mode==Mode.NORMAL:
            if self.phase==Phase.LIFT:
                xyz=self._reach_xyz+self._reach_xyz.new_tensor((0,0,d.lift_m))
            else:
                xyz=self._reach_xyz+self._reach_xyz.new_tensor((-d.retract_m,0,d.lift_m))
            return xyz,(self._palm_quat if d.normal_hold_orientation else None)
        if self.phase==Phase.DESCEND:
            xyz=self._palm_xyz+self._palm_xyz.new_tensor((0,0,-d.descent_m))
        elif self.phase==Phase.LIFT:
            xyz=self._palm_xyz+self._palm_xyz.new_tensor((0,0,d.lift_m))
        else:
            xyz=self._palm_xyz+self._palm_xyz.new_tensor((-d.retract_m,0,d.lift_m))
        return xyz,self._palm_quat

    def _ik_step(self,dt):
        xyz_goal,quat_goal=self._cartesian_target()
        xyz,quat=self.wrist_pose_torso()
        J=self.jacobian_6d_torso()
        dp=xyz_goal-xyz
        if quat_goal is not None:
            d_rot=quat_error_axis_angle(quat_goal,quat)
            weight=self.cfg.orientation_weight_m_per_rad
            Juse=torch.cat((J[:3],weight*J[3:]),dim=0)
            err=torch.cat((dp,weight*d_rot))
        else:
            Juse=J[:3]; err=dp
        # Damped least squares with numerically stable small task-space solve.
        D=Juse.shape[0]
        mat=Juse@Juse.T+self.cfg.damping**2*torch.eye(D,dtype=self.dtype,device=self.device)
        dq=Juse.T@torch.linalg.solve(mat,err)
        # Redundancy preference, projected into the task's null space.
        preference=torch.zeros(7,dtype=self.dtype,device=self.device)
        preference[5]=-self.cfg.nullspace_gain*(
            self.arm_q()[5]-self.cfg.wrist_pitch_preference_rad)
        # Damped projector limits twitching near singularities.
        damped_pinv=Juse.T@torch.linalg.solve(mat,torch.eye(D,dtype=self.dtype,device=self.device))
        dq += (torch.eye(7,dtype=self.dtype,device=self.device)-damped_pinv@Juse)@preference
        dq=dq.clamp(-self.cfg.max_increment_rad,self.cfg.max_increment_rad)
        proposal=self._last_cmd+dq
        # No desired-stage q reset; limit incremental setpoint change smoothly.
        return proposal

    def _fail(self,msg):
        self.phase=Phase.FAILED;self.reason=str(msg)
        # Host must keep balancing, choose a safe hold/stop; no blind HOME.
        return self.joint_ids,self.arm_q(),self.phase

    def step(self,dt:float):
        """Return (arm joint IDs, target q[7], phase). Never writes actions.

        REACH tolerance is *diagnostic* and does NOT cause a crash.
        Unsafe joint limits/collisions, missing sensor or bad math do stop.
        """
        if not math.isfinite(dt) or dt<=0: raise ValueError('dt must be >0')
        if self.phase in (Phase.IDLE,Phase.DONE,Phase.FAILED):
            return self.joint_ids,self.arm_q(),self.phase
        try:
            measured=self._check_q(self.arm_q())
            self.elapsed+=dt
            duration=self._duration()
            if self._goal is not None:
                u=min(1.,self.elapsed/duration)
                smooth=u*u*(3-2*u)
                proposal=self._stage_start+smooth*(self._goal-self._stage_start)
            else:
                proposal=self._ik_step(dt)
            max_step=self.cfg.max_speed_rad_s*dt
            cmd=self._last_cmd+(proposal-self._last_cmd).clamp(-max_step,max_step)
            self._check_q(cmd)
            if not self._safety(measured,cmd,self.phase):
                return self._fail(f'Unsafe/unknown transition: {self.phase.value}')
            self._last_cmd=cmd.detach().clone()
            # q stages finish only when interpolated target is truly commanded.
            if self.elapsed>=duration:
                done=(self._goal is None or
                      float(torch.max(torch.abs(self._goal-cmd)))<=1e-3)
                if done or self.elapsed>=duration+self.cfg.stage_extra_s:
                    if not done and self._goal is not None:
                        return self._fail(f'Joint goal not commanded within timeout: {self.phase.value}')
                    if self.phase==Phase.REACH:
                        real_xyz,real_q=self.wrist_pose_torso()
                        self.reach_error_m=float(torch.linalg.vector_norm(self._reach_xyz-real_xyz))
                        if self._reach_quat is not None:
                            self.orientation_error_rad=float(torch.linalg.vector_norm(
                                quat_error_axis_angle(self._reach_quat,real_q)))
                    self._advance()
            return self.joint_ids,cmd,self.phase
        except (RuntimeError,ValueError,IndexError) as exc:
            return self._fail(str(exc))
