# S.H.O.P. — G1 left-arm manipulation port to Isaac Lab

**Handoff / integration source code, NOT a tested drop-in scene. SIMULATION ONLY.**

## What is included

- `g1_arm_isaaclab.py`: modular left-arm command generator. Does **not** load MuJoCo, use DDS, create a simulator, move other joints, or run hardware.
- `INTEGRATION_EXAMPLE.py`: annotated snippet for the teammate's *existing* Isaac Lab simulation loop.
- `original_arm_targets.py`, `original_arm_ik.py`, `original_mujoco_arm_ik_dynamic_lift.py`: original references, unchanged.

## Original implementation / port

- G1 29-DoF, left arm 7 joints. Original DDS motor IDs 15–21 **must not be reused as Isaac Lab indices**. The module finds joints by name.
- Original IK: translational 3×7 Jacobian, Damped Least Squares with damping **0.05**, up to 0.05 rad per correction, 5 mm original offline convergence criterion. The port implements **online incremental DLS**, rate-limited, 10 mm measured wrist target tolerance by default. It is *not* a bit-for-bit reproduction of offline MuJoCo IK.
- End-effector: `left_wrist_yaw_link` (wrist, **not** hand/fingertip TCP). Targets in meters **relative to `torso_link`**, not world coordinates.
- Preserved exact PRE_GRASP: `[-0.25, 0, 0, 0.10, 0, -0.05, 0]` rad in shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw order.
- Stages: measured HOME → PRE_GRASP → REACH → LIFT (+0.05 m torso Z) → RETRACT (−0.10 m torso X) → PRE_GRASP → measured HOME.
- Smoothstep joint interpolation for posture stages. Cartesian feedback DLS for REACH/LIFT/RETRACT. Original MuJoCo reference REACH joint pose is included as `REACH_REFERENCE`, but **cannot be turned into an Isaac Lab XYZ target without the correct model's forward kinematics**.

## Integrate in teammate's Isaac Lab project

1. Copy `g1_arm_isaaclab.py` into a Python-importable folder in the teammate's project.
2. Ensure G1 articulation exists and is initialized. Its joints must have the names `left_shoulder_pitch_joint`, `left_shoulder_roll_joint`, `left_shoulder_yaw_joint`, `left_elbow_joint`, `left_wrist_roll_joint`, `left_wrist_pitch_joint`, `left_wrist_yaw_joint`; and bodies `torso_link`, `left_wrist_yaw_link`. If their USD uses other names, adapt `_find_arm_joints()` / `_find_body()`.
3. Construct `G1LeftArmController(robot, env_id=0)` after initialization. Call `arm.start([x, y, z])` **only after obtaining a reachable wrist target in torso coordinates for their USD**.
4. On each physics/control step, after the host has refreshed robot state, call `ids, q, phase = arm.step(dt)`. Merge `q` into the existing **whole-body joint-position targets** at indices `ids` for environment 0. The host retains all other joint commands and is responsible for calling `robot.set_joint_position_target(...)`, stepping simulation, and updating articulation state. See `INTEGRATION_EXAMPLE.py`.
5. Validate arm indices, Jacobian body/column layout, quaternion ordering, target frame, actuator control mode and gains **in their installed Isaac Lab version**. Ensure the host's locomotion policy does not overwrite the arm target afterward; if it does, coordinate the action merging.
6. Begin with a static, collision-free, supported robot and a reachable target; test one stage at a time. Use only in simulation until thoroughly validated. The module does not perform collision checking or balance control.

## Important limitations / TODO for teammate

- **Untested against actual Isaac Lab/Isaac Sim/PhysX** (not installed in this environment). Code is syntactically checked only. Actual USD joint names, body names, Jacobian indexing and simulation loop need verification.
- `root_physx_view.get_jacobians()` layout is assumed to follow common Isaac Lab articulation layout: floating base has 6 extra generalized coordinates and excludes root body Jacobian. Validate with your specific G1 articulation.
- Uses `robot.data.body_pos_w`, `body_quat_w`, `soft_joint_pos_limits`, `joint_pos` and `root_physx_view`. Verify that they exist in your Isaac Lab release.
- Uses torso-relative translational Jacobian; does not solve wrist orientation. Assumes left-arm joint motion does not rotate `torso_link` directly.
- Does not calculate FK for original REACH pose, nor import MuJoCo's XML/USD. **Target XYZ must be provided by teammate/vision or measured using their Isaac Lab robot**.
- Does not control gripper, object attachment, contact/collision avoidance, dynamics, whole-body stability, or robot hardware.
- It automatically returns after RETRACT as in the original test. **Do not use with a held object without an appropriate release/placement sequence**.
- Joint limits are clipped to Isaac Lab *soft* limits; this does not guarantee collision safety. Stage timeouts report `FAILED`; host must implement its own safe stop.
- For the original MuJoCo test, REACH ~40–43 mm, LIFT ~48 mm, RETRACT ~22 mm tracking errors were observed; these have **not** been fixed or validated in Isaac Lab.
- Single environment (`env_id`) per controller instance. `dt` is the time between actual calls to `step`, not assumed to be 0.002 seconds.

## How to obtain a REACH target

Preferred: teammate provides a torso-frame wrist target from their Isaac Lab FK or perception pipeline. To convert a world-frame point to torso coordinates, use `R_torso.T @ (point_world - torso_position_world)`. For object picking, convert from object pose to desired **wrist** pose using the hand/TCP offset first. **Do not feed object-center coordinates as wrist target without that offset.**

## Source fidelity

The original `arm_targets.py` has a `HOME` configuration but the final MuJoCo script starts/returns to the **measured current arm posture**, not that constant. This port does the same. The original final script has an extraneous final line `un_stage(...)`; it is preserved in the reference copy but not ported.
