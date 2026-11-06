"""Unitree G1 (29 body joints) fitted with Inspire 5-finger hands (model RH56DFQ).

Place this file at:
  source/unitree_rl_lab/unitree_rl_lab/assets/robots/g1_inspire.py

It reuses Unitree's G1 config (same body joints, actuators, start pose) and only
swaps the robot model for the URDF with Inspire hands from unitree_ros, adding
actuators for the finger joints.

Inspire hand joints (per hand, prefix L_ or R_):
  6 driven ("active") joints:  thumb_proximal_yaw, thumb_proximal_pitch,
                               index_proximal, middle_proximal, ring_proximal, pinky_proximal
  6 linked ("mimic") joints:   thumb_intermediate (1.6 x thumb pitch), thumb_distal (2.4 x thumb pitch),
                               index/middle/ring/pinky_intermediate (1.0 x their proximal joint)
On the real hand the linked joints are mechanically coupled. In simulation we command
them explicitly with the same ratios (see hand_joint_targets below).
"""

import os

from isaaclab.actuators import ImplicitActuatorCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG, UNITREE_ROS_DIR, UnitreeUrdfFileCfg

INSPIRE_URDF_NAME = "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"

# The 29 body joints the walking policy was trained on (Unitree SDK order).
G1_BODY_JOINTS: list[str] = list(UNITREE_G1_29DOF_CFG.joint_sdk_names)

# Inspire hand joints. "Leader" joints are motor-driven; "followers" copy a leader times a ratio.
INSPIRE_LEADERS = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]
INSPIRE_FOLLOWERS = {  # follower: (leader, multiplier), from the URDF <mimic> tags
    "thumb_intermediate_joint": ("thumb_proximal_pitch_joint", 1.6),
    "thumb_distal_joint": ("thumb_proximal_pitch_joint", 2.4),
    "index_intermediate_joint": ("index_proximal_joint", 1.0),
    "middle_intermediate_joint": ("middle_proximal_joint", 1.0),
    "ring_intermediate_joint": ("ring_proximal_joint", 1.0),
    "pinky_intermediate_joint": ("pinky_proximal_joint", 1.0),
}
INSPIRE_HAND_JOINT_REGEX = ["[LR]_.*_joint"]

# Leader angles (rad) for a fully open and a closed power grasp. Joint limits from the URDF:
#   thumb_proximal_yaw -0.1..1.3, thumb_proximal_pitch -0.1..0.6, fingers 0..1.7
HAND_OPEN = {name: 0.0 for name in INSPIRE_LEADERS}
HAND_CLOSED = {
    "thumb_proximal_yaw_joint": 1.1,  # swing the thumb across the palm
    "thumb_proximal_pitch_joint": 0.45,
    "index_proximal_joint": 1.4,
    "middle_proximal_joint": 1.4,
    "ring_proximal_joint": 1.4,
    "pinky_proximal_joint": 1.4,
}


def hand_joint_targets(side: str, closure: float) -> dict[str, float]:
    """Target angle for every joint of one hand.

    side: "L" or "R".  closure: 0.0 = fully open, 1.0 = fully closed (HAND_CLOSED).
    """
    closure = max(0.0, min(1.0, closure))
    targets = {}
    for leader in INSPIRE_LEADERS:
        targets[f"{side}_{leader}"] = HAND_OPEN[leader] + closure * (HAND_CLOSED[leader] - HAND_OPEN[leader])
    for follower, (leader, ratio) in INSPIRE_FOLLOWERS.items():
        targets[f"{side}_{follower}"] = ratio * targets[f"{side}_{leader}"]
    return targets


def _find_inspire_urdf() -> str:
    """Look for the Inspire URDF next to the G1 URDF you already use, then under UNITREE_ROS_DIR."""
    candidates = []
    base_path = getattr(UNITREE_G1_29DOF_CFG.spawn, "asset_path", None)
    if base_path:
        candidates.append(os.path.join(os.path.dirname(base_path), INSPIRE_URDF_NAME))
    candidates.append(os.path.join(UNITREE_ROS_DIR, "robots", "g1_description", INSPIRE_URDF_NAME))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {INSPIRE_URDF_NAME}. Looked in:\n  " + "\n  ".join(candidates)
        + "\nIt ships with unitree_ros in robots/g1_description/ (run `git pull` in unitree_ros if it's missing)."
    )


G1_INSPIRE_CFG = UNITREE_G1_29DOF_CFG.replace(
    spawn=UnitreeUrdfFileCfg(asset_path=_find_inspire_urdf()),
    actuators={
        **UNITREE_G1_29DOF_CFG.actuators,
        # Inspire fingers: small, light, slow motors (URDF effort limit 1 N*m).
        "inspire_hand": ImplicitActuatorCfg(
            joint_names_expr=INSPIRE_HAND_JOINT_REGEX,
            effort_limit_sim=1.0,
            velocity_limit_sim=3.0,
            stiffness=3.0,
            damping=0.1,
            armature=0.001,
        ),
    },
)
