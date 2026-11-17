"""EXAMPLE FRAGMENT ONLY — paste inside teammate's existing Isaac Lab simulation loop.

Do not execute as standalone script: `robot`, `dt`, and `whole_body_targets`
come from the host's already-running simulation / locomotion controller.
"""
from g1_arm_isaaclab import G1LeftArmController, Phase

# ONCE after the G1 Articulation has been initialized:
arm = G1LeftArmController(robot, env_id=0)
# A target in meters, expressed in torso_link frame, for the WRIST link.
# REPLACE with a verified reachable target from the Isaac Lab robot / vision.
# arm.start([x_torso, y_torso, z_torso])

# INSIDE THE HOST CONTROL LOOP, after robot state has been refreshed:
# joint_ids, arm_targets, phase = arm.step(dt)
# whole_body_targets is the host's EXISTING command tensor [num_envs, num_joints]
# (do not construct it from measured positions if locomotion needs its targets).
# whole_body_targets[0, joint_ids] = arm_targets
# robot.set_joint_position_target(whole_body_targets)
# host sim.step() and robot.update(dt) are unchanged.
# if phase == Phase.FAILED:
#     print('Manipulation failed:', arm.failure_reason)
#     # Host decides how to stop/hold safely; do not blindly keep commanding.
