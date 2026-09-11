"""Scripted pick/place primitives built on the IK solver.

These are the low-level "hands" the VLA planner's high-level plan gets
executed through: the planner only ever says {arm, action, object|target_xyz},
never a joint angle.
"""

from __future__ import annotations

import numpy as np

from aisummit.control.ik import solve_ik
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
    env: DinnerTableEnv, ctrl: np.ndarray, side: str, target_xyz: np.ndarray, target_quat: np.ndarray
) -> tuple[np.ndarray, Observation]:
    """Orientation-aware counterpart to `move_to`, used by the grasp
    pipeline for elongated objects. `move_to` is untouched and still used
    for radially symmetric objects -- see grasping/planner.py."""
    angles = solve_ik(env.model, env.data, side, np.asarray(target_xyz), target_quat=target_quat)
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
    lift_height: float = 0.12,
) -> tuple[np.ndarray, Observation]:
    """Same shape as `pick`, but drives a specific grasp pose (position +
    orientation) instead of always approaching an object's CoM from
    whatever orientation position-only IK happens to converge to.

    Deliberately single-stage (no separate high-approach waypoint before
    descending, unlike `pick`): empirically, adding one for the oriented
    case made convergence much *worse* (3.6cm final position error with a
    direct move vs. 14cm going through an intermediate waypoint first) --
    the twisted wrist orientation combined with an intermediate pose seeds
    the second IK solve into a different, poorly-conditioned local solution
    (a kinematic "elbow flip"). This trades away the side-swipe protection
    a high approach gives `pick()`; `grasping/collision.py` covers the
    final pose, not the swept path, which is an accepted gap for the
    current small demo scene -- see README known limitations."""
    target_xyz = np.asarray(target_xyz)
    ctrl = set_gripper(env, ctrl, side, GRIPPER_OPEN)
    ctrl, obs = move_to_pose(env, ctrl, side, target_xyz, target_quat)
    ctrl = set_gripper(env, ctrl, side, GRIPPER_CLOSED)
    obs = hold(env, ctrl)
    ctrl, obs = move_to_pose(env, ctrl, side, target_xyz + np.array([0, 0, lift_height]), target_quat)
    return ctrl, obs


def place_oriented(
    env: DinnerTableEnv, ctrl: np.ndarray, side: str, target_xyz: np.ndarray, target_quat: np.ndarray,
    retreat_height: float = 0.10,
) -> tuple[np.ndarray, Observation]:
    """Orientation-preserving counterpart to `place` -- keeps the same
    grasp orientation through descent (requirement: predictable object
    orientation during transport, particularly for knives). Single-stage
    descent for the same reason as `pick_oriented`: an intermediate
    waypoint measurably hurt IK convergence for a twisted orientation."""
    target_xyz = np.asarray(target_xyz)
    ctrl, obs = move_to_pose(env, ctrl, side, target_xyz, target_quat)
    ctrl = set_gripper(env, ctrl, side, GRIPPER_OPEN)
    obs = hold(env, ctrl, steps=30)
    ctrl, obs = move_to_pose(env, ctrl, side, target_xyz + np.array([0, 0, retreat_height]), target_quat)
    return ctrl, obs
