"""Scripted pick/place primitives built on the IK solver.

These are the low-level "hands" the VLA planner's high-level plan gets
executed through: the planner only ever says {arm, action, object|target_xyz},
never a joint angle.
"""

from __future__ import annotations

import mujoco
import numpy as np

from aisummit.control.ik import finger_target_to_site_target, solve_ik
from aisummit.sim.env import ARM_JOINTS, DinnerTableEnv, Observation

GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.002
_SETTLE_STEPS = 300
# Oriented moves reach more twisted joint configurations than the top-down,
# minimal-orientation poses position-only `move_to` was tuned against --
# the arm's position-actuator PD gains (aloha.xml's fixed kp per joint)
# settle those far slower under gravity. Empirically, ~300 steps left a
# 6.5cm gripper-position error that was still visibly converging, not
# stuck; ~1200 gets it consistently under 1cm. Only `move_to_pose` uses
# this -- `move_to` (and therefore the existing radial-object grasp path)
# is untouched.
_ORIENTED_SETTLE_STEPS = 1200
_SUBSTEPS = 5
_HOLD_STEPS = 40


def _set_arm_target(env: DinnerTableEnv, ctrl: np.ndarray, side: str, joint_angles: np.ndarray) -> np.ndarray:
    for j, joint_name in enumerate(ARM_JOINTS):
        ctrl[env.actuator_index(side, joint_name)] = joint_angles[j]
    return ctrl


def _current_arm_angles(env: DinnerTableEnv, side: str) -> np.ndarray:
    """The arm's real, physically-settled joint angles right now -- used to
    warm-start a follow-up solve from wherever the arm actually is, rather
    than from the original (now stale) seed. More reliable than reusing an
    earlier IK solution: this reflects what actually happened under gravity
    and the position actuators' PD settling, not just the kinematic target."""
    return np.array(
        [env.data.qpos[env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{j}")]]
         for j in ARM_JOINTS]
    )


def set_gripper(env: DinnerTableEnv, ctrl: np.ndarray, side: str, opening: float) -> np.ndarray:
    ctrl[env.actuator_index(side, "gripper")] = opening
    return ctrl


def move_to(
    env: DinnerTableEnv, ctrl: np.ndarray, side: str, target_xyz: np.ndarray
) -> tuple[np.ndarray, Observation]:
    angles = solve_ik(env.model, env.data, side, np.asarray(target_xyz))
    ctrl = _set_arm_target(env, ctrl, side, angles)
    obs = None
    for _ in range(_SETTLE_STEPS // _SUBSTEPS):
        obs = env.step(ctrl, n_substeps=_SUBSTEPS)
    return ctrl, obs


def move_to_pose(
    env: DinnerTableEnv, ctrl: np.ndarray, side: str, target_xyz: np.ndarray, target_quat: np.ndarray,
    seed_angles: np.ndarray | None = None,
) -> tuple[np.ndarray, Observation]:
    """Orientation-aware counterpart to `move_to`, used by the grasp
    pipeline for elongated objects. `move_to` is untouched and still used
    for radially symmetric objects -- see grasping/planner.py.

    `seed_angles` warm-starts the IK solve from a specific configuration
    instead of the arm's actual current pose -- needed for far/twisted
    targets where cold-start damped least-squares can fail to converge
    even though a solution exists nearby (see solve_ik's docstring)."""
    angles = solve_ik(
        env.model, env.data, side, np.asarray(target_xyz), target_quat=target_quat, seed_angles=seed_angles
    )
    ctrl = _set_arm_target(env, ctrl, side, angles)
    obs = None
    for _ in range(_ORIENTED_SETTLE_STEPS // _SUBSTEPS):
        obs = env.step(ctrl, n_substeps=_SUBSTEPS)
    return ctrl, obs


def hold(env: DinnerTableEnv, ctrl: np.ndarray, steps: int = _HOLD_STEPS) -> Observation:
    obs = None
    for _ in range(steps):
        obs = env.step(ctrl, n_substeps=_SUBSTEPS)
    return obs


def pick(
    env: DinnerTableEnv, ctrl: np.ndarray, side: str, object_name: str,
    approach_height: float = 0.10, grasp_height: float = 0.02, lift_height: float = 0.12,
) -> tuple[np.ndarray, Observation]:
    obj_pos = env.body_xpos(object_name)
    ctrl = set_gripper(env, ctrl, side, GRIPPER_OPEN)
    ctrl, obs = move_to(env, ctrl, side, obj_pos + np.array([0, 0, approach_height]))
    ctrl, obs = move_to(env, ctrl, side, obj_pos + np.array([0, 0, grasp_height]))
    ctrl = set_gripper(env, ctrl, side, GRIPPER_CLOSED)
    obs = hold(env, ctrl)
    ctrl, obs = move_to(env, ctrl, side, obj_pos + np.array([0, 0, lift_height]))
    return ctrl, obs


def place(
    env: DinnerTableEnv, ctrl: np.ndarray, side: str, target_xyz: np.ndarray,
    descend_height: float = 0.10,
) -> tuple[np.ndarray, Observation]:
    target_xyz = np.asarray(target_xyz)
    ctrl, obs = move_to(env, ctrl, side, target_xyz + np.array([0, 0, descend_height]))
    ctrl, obs = move_to(env, ctrl, side, target_xyz)
    ctrl = set_gripper(env, ctrl, side, GRIPPER_OPEN)
    obs = hold(env, ctrl, steps=30)
    ctrl, obs = move_to(env, ctrl, side, target_xyz + np.array([0, 0, descend_height]))
    return ctrl, obs


def pick_oriented(
    env: DinnerTableEnv, ctrl: np.ndarray, side: str, target_xyz: np.ndarray, target_quat: np.ndarray,
    lift_height: float = 0.12, seed_angles: np.ndarray | None = None,
    final_target_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, Observation]:
    """Same shape as `pick`, but drives a specific grasp pose (position +
    orientation) instead of always approaching an object's CoM from
    whatever orientation position-only IK happens to converge to.

    Deliberately single-stage for the APPROACH (no separate high-approach
    waypoint before descending, unlike `pick`): empirically, adding one for
    the oriented case made convergence much *worse* (3.6cm final position
    error with a direct move vs. 14cm going through an intermediate
    waypoint first) -- the twisted wrist orientation combined with an
    intermediate pose seeds the second IK solve into a different,
    poorly-conditioned local solution (a kinematic "elbow flip"). This
    trades away the side-swipe protection a high approach gives `pick()`;
    `grasping/collision.py` covers the final (post-descent, if any) pose,
    not the swept path, which is an accepted gap for the current small
    demo scene -- see README known limitations.

    `final_target_xyz`, if given, adds a SECOND, translation-only descent
    after the approach: same `target_quat`, warm-started from the arm's own
    just-settled joint angles (not the original `seed_angles`) rather than
    a fresh orientation search. This exists because `target_xyz` (the
    approach point) usually isn't the real close height -- candidates from
    `grasping/candidates.py` add a clearance margin above the object so the
    gripper body clears the table at wide wrist angles, which also means
    the fingers close well above the object unless something brings them
    back down. A two-waypoint POSITION-only move is safe here in a way a
    two-waypoint POSITION+ORIENTATION move (tried and rejected above) is
    not: the hard part -- finding a valid orientation -- is already solved
    by the time this step runs, and warm-starting from the arm's actual
    current configuration (already achieving that exact orientation) keeps
    the second solve local rather than risking a fresh elbow flip.

    `target_xyz` (and `final_target_xyz`) mean the true intended
    FINGER-MIDPOINT position (the grasp point), not the `{side}/gripper`
    site `move_to_pose` itself moves -- `finger_target_to_site_target`
    converts between the two (see control/ik.py's docstring for why they
    aren't the same point).

    `seed_angles`, if given (grasping/candidates.py always provides one),
    warm-starts the first solve -- a cold start from the arm's resting pose
    can fail to converge for a far, heavily-twisted target even when one is
    reachable (found via a real false positive during development: a
    candidate judged collision-free and well-aligned was actually nowhere
    near its target, because nothing had checked the solved pose's
    position error)."""
    target_xyz = np.asarray(target_xyz)
    ctrl = set_gripper(env, ctrl, side, GRIPPER_OPEN)
    ctrl, obs = move_to_pose(
        env, ctrl, side, finger_target_to_site_target(target_xyz, target_quat), target_quat,
        seed_angles=seed_angles,
    )

    close_point = target_xyz
    if final_target_xyz is not None:
        close_point = np.asarray(final_target_xyz)
        descent_seed = _current_arm_angles(env, side)
        ctrl, obs = move_to_pose(
            env, ctrl, side, finger_target_to_site_target(close_point, target_quat), target_quat,
            seed_angles=descent_seed,
        )

    ctrl = set_gripper(env, ctrl, side, GRIPPER_CLOSED)
    obs = hold(env, ctrl)
    lift_seed = _current_arm_angles(env, side) if final_target_xyz is not None else seed_angles
    lift_target = close_point + np.array([0, 0, lift_height])
    ctrl, obs = move_to_pose(
        env, ctrl, side, finger_target_to_site_target(lift_target, target_quat), target_quat,
        seed_angles=lift_seed,
    )
    return ctrl, obs


def place_oriented(
    env: DinnerTableEnv, ctrl: np.ndarray, side: str, target_xyz: np.ndarray, target_quat: np.ndarray,
    retreat_height: float = 0.10,
) -> tuple[np.ndarray, Observation]:
    """Orientation-preserving counterpart to `place` -- keeps the same
    grasp orientation through descent (requirement: predictable object
    orientation during transport, particularly for knives). Single-stage
    descent for the same reason as `pick_oriented`: an intermediate
    waypoint measurably hurt IK convergence for a twisted orientation.
    `target_xyz` means the finger-midpoint target, same convention as
    `pick_oriented` -- see its docstring."""
    target_xyz = np.asarray(target_xyz)
    ctrl, obs = move_to_pose(env, ctrl, side, finger_target_to_site_target(target_xyz, target_quat), target_quat)
    ctrl = set_gripper(env, ctrl, side, GRIPPER_OPEN)
    obs = hold(env, ctrl, steps=30)
    retreat_target = target_xyz + np.array([0, 0, retreat_height])
    ctrl, obs = move_to_pose(env, ctrl, side, finger_target_to_site_target(retreat_target, target_quat), target_quat)
    return ctrl, obs
