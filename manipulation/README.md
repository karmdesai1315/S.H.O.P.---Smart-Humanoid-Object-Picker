# S.H.O.P. — Unitree G1 Manipulation Controller (Isaac Lab)

**Status:** Integration-ready source code; **not yet tested in Isaac Lab**. Simulation only.

This folder contains the independent **7-DoF left-arm manipulation controller** developed for S.H.O.P. (Smart Humanoid Object Picker). It is intended to be integrated into the team's existing Unitree G1 Isaac Lab simulation without replacing the locomotion or balance controller.

## Files

| File | Purpose |
| --- | --- |
| `g1_arm_isaaclab.py` | Main arm command generator: joint mapping, torso-frame wrist position, Jacobian-based DLS IK, smooth trajectories, staged state machine. |
| `INTEGRATION_EXAMPLE.py` | **Illustrative integration fragment**, not an executable standalone simulation. |
| `README.md` | Integration notes, expected behavior, limitations and validation checklist. |

**No MuJoCo, DDS, or Unitree SDK is required by these files.** The controller uses Python and PyTorch, and expects a robot articulation provided by the host Isaac Lab project.

## Architecture and responsibility

The controller **does not** create a scene, start Isaac Lab, step the simulator, or write commands to the robot by itself. On each host control tick it returns the **seven left-arm joint indices**, their **position targets**, and the current **phase**. The host must merge those targets with its existing whole-body commands, retain control of all other joints, and apply them through its own Isaac Lab loop.

The locomotion controller must not overwrite these same seven joints after merging. Coordinate command ownership explicitly; do not replace the whole-body action with a seven-joint action.

## Implemented behavior

- **Robot:** Unitree G1 (original development model: 29 actuated joints).
- **Controlled joints:** left shoulder pitch/roll/yaw, left elbow, left wrist roll/pitch/yaw.
- **End-effector reference:** `left_wrist_yaw_link` (the **wrist link**, not a gripper fingertip or tool center point).
- **Reference frame:** all target XYZ values are **meters relative to `torso_link`**.
- **IK:** incremental translational **3 × 7 Jacobian**, Damped Least Squares (DLS), default damping `0.05`; joint correction and velocity limits.
- **Posture transitions:** smoothstep interpolation.
- **Feedback:** measured wrist position and measured joint positions from the articulation.
- **Safety-related guards:** soft joint-position clipping, phase timeouts, finite target validation, and a `FAILED` state. These are **not** collision avoidance or safety certification.

### Automatic sequence

After the host calls `arm.start(reach_xyz_torso=[x, y, z])`, subsequent calls to `arm.step(dt)` progress through:

`PRE_GRASP → REACH → LIFT → RETRACT → RETURN_PRE_GRASP → RETURN_HOME → DONE`

The controller saves the **measured starting arm posture** as HOME; it does not first move to a fixed HOME configuration.

- **PRE_GRASP:** predefined seven-joint posture.
- **REACH:** moves the wrist toward the supplied torso-frame XYZ.
- **LIFT:** targets **+0.05 m** in torso Z relative to REACH.
- **RETRACT:** targets **−0.10 m** in torso X, while retaining the +0.05 m Z offset.
- **RETURN:** goes through PRE_GRASP and returns to the recorded starting arm posture.

The default wrist tolerance is **0.010 m**. Each phase must remain within tolerance for the configured settling time before advancing; a phase can fail on timeout. This is a **position-only** controller: it does not command wrist orientation, close a hand, grasp an object, or place an object.

## Integration in the existing Isaac Lab project

1. Keep the work on branch **`feature/g1-manipulation`** until integration and validation are complete.
2. Place `g1_arm_isaaclab.py` in a Python-importable location in the existing project.
3. Confirm that the G1 articulation is initialized before constructing the controller. Verify its exact joint and body names, joint limits, quaternion conventions, Jacobian layout, and control mode against the **installed Isaac Lab version and the team's USD asset**.
4. Create a controller instance **once** for the relevant simulation environment:

   ```python
   from g1_arm_isaaclab import G1LeftArmController, Phase

   arm = G1LeftArmController(robot, env_id=0)
   ```

5. Supply a **verified reachable wrist target** in torso coordinates and start the sequence:

   ```python
   arm.start(reach_xyz_torso=[x_torso, y_torso, z_torso])
   ```

   The three variables above are placeholders, **not** a verified target. The controller does not select a target automatically.

6. In the host's existing control loop, **after articulation state is refreshed**, merge the controller's outputs into the host's existing whole-body joint-position target tensor:

   ```python
   joint_ids, q_arm, phase = arm.step(dt)
   whole_body_targets[0, joint_ids] = q_arm
   robot.set_joint_position_target(whole_body_targets)
   ```

   The host remains responsible for simulation stepping, state updates, actuator settings, locomotion actions, and failure handling. Adapt the snippet to the actual project API and control architecture. `INTEGRATION_EXAMPLE.py` is a guide, **not** a standalone runnable entry point.

7. If `phase == Phase.FAILED`, inspect `arm.failure_reason` and use the host's safe stop/hold behavior. Do not assume the module provides collision or balance recovery.

### Target coordinates: torso vs. world

A constant torso-frame XYZ **moves with the robot** in the world. It is suitable for initial arm tests, but it does **not** represent a fixed object location in the map.

To reach a fixed world-frame target, the host must update the target relative to the torso using the robot's current torso pose:

```text
p_target_torso = R_torso_world.T @ (p_target_world - p_torso_world)
```

For actual object pickup, compute a desired **wrist** target from the object pose and the hand/tool offset. Do **not** assume that the object's center is the correct wrist target.

The present implementation accepts the target at `start()`; it does **not** continuously refresh a world-fixed target as the robot walks. Test with a supported, stationary robot first. Dynamic world-target tracking would require additional integration or controller changes.

## Assumptions that the teammate must verify

- Exact G1 joint names are expected to be `left_shoulder_pitch_joint`, `left_shoulder_roll_joint`, `left_shoulder_yaw_joint`, `left_elbow_joint`, `left_wrist_roll_joint`, `left_wrist_pitch_joint`, and `left_wrist_yaw_joint`.
- Required body names: `torso_link` and `left_wrist_yaw_link`.
- Original MuJoCo/DDS motor IDs **15–21 are not Isaac Lab joint indices**; this controller resolves joint indices by name.
- The code assumes an Isaac Lab articulation exposing `robot.data.joint_pos`, `body_pos_w`, `body_quat_w`, `soft_joint_pos_limits`, and `robot.root_physx_view.get_jacobians()`.
- Verify the PhysX Jacobian body-row and joint-column offsets, especially for floating-base G1 models.
- Verify that the torso quaternion is ordered **[w, x, y, z]** and that torso-relative Jacobians match the actual articulation.
- The host must use compatible joint-position actuation and ensure locomotion commands do not conflict with arm commands.
- One controller instance controls one environment (`env_id`). Pass the **actual interval between `step()` calls** as `dt`.

## Validation checklist (simulation only)

- [ ] Import the module in the team's installed Isaac Lab environment.
- [ ] Resolve all seven arm joints and both body links correctly.
- [ ] Verify wrist FK/torso-frame position and Jacobian response to small joint changes.
- [ ] Confirm actuator gains, limits, and command merging with the locomotion controller.
- [ ] Test PRE_GRASP with the robot supported and stationary.
- [ ] Select a reachable, collision-free wrist XYZ **from the Isaac Lab asset**, then test REACH.
- [ ] Measure actual wrist tracking error and check the 5 cm LIFT and 10 cm RETRACT offsets.
- [ ] Check self-collisions, environment collisions, torque/speed behavior, balance, and phase timeout handling.
- [ ] Only then test simultaneous locomotion and manipulation.
- [ ] Open a Pull Request after successful integration and review; do not merge untested behavior into `main`.

## Known limitations

- **Not run or validated in Isaac Lab/Isaac Sim/PhysX**; syntax and source structure alone do not prove runtime compatibility.
- **No vision or automatic target detection.** A torso-frame wrist XYZ must be supplied by the host.
- **No grasping or hand control.** LIFT is a wrist-motion test, not proof of lifting an object.
- **No collision planning, obstacle avoidance, balance controller, or hardware support.**
- **No orientation IK** or gripper/tool-center-point compensation.
- **No guarantee of reaching every supplied XYZ**; reachability, kinematics and phase success must be checked in simulation.
- The original MuJoCo experiments exhibited nonzero tracking errors; these results have **not** been resolved or reproduced in Isaac Lab.

## Ownership

**Manipulation:** independent left-arm controller, targets, IK, and staged motions.  
**Locomotion / simulation:** host Isaac Lab environment, whole-body control, stepping, action arbitration, and integration testing.  
**Perception (future):** object localization and conversion to a valid wrist target.

**This branch is a handoff for integration and testing, not a claim of a working autonomous object-picking system.**
