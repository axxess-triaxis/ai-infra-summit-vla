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
