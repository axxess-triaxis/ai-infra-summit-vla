"""Damped least-squares numerical IK for one ALOHA arm.

Solves position-only IK (no orientation term) against the `{side}/gripper`
site's 6 arm joints. Runs on a scratch copy of MjData so it never disturbs
the live simulation while it iterates.
"""

from __future__ import annotations

import mujoco
import numpy as np

from aisummit.sim.env import ARM_JOINTS

_MAX_ITERS = 150
_DAMPING = 0.05
_STEP_SIZE = 0.5
_TOLERANCE = 1e-3


def _arm_dof_indices(model: mujoco.MjModel, side: str) -> list[int]:
    indices = []
    for joint_name in ARM_JOINTS:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{joint_name}")
        indices.append(model.jnt_dofadr[jnt_id])
    return indices


def _joint_ranges(model: mujoco.MjModel, side: str) -> np.ndarray:
    ranges = []
    for joint_name in ARM_JOINTS:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{joint_name}")
        ranges.append(model.jnt_range[jnt_id])
    return np.array(ranges)


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    target_xyz: np.ndarray,
) -> np.ndarray:
    """Returns the 6 target joint angles for `side`'s arm that bring its
    gripper site closest to `target_xyz`. Does not mutate `data`."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = data.qvel
    mujoco.mj_forward(model, scratch)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/gripper")
    dof_idx = _arm_dof_indices(model, side)
    qpos_idx = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{j}")]
                for j in ARM_JOINTS]
    joint_ranges = _joint_ranges(model, side)

    jacp = np.zeros((3, model.nv))
    for _ in range(_MAX_ITERS):
        site_pos = scratch.site(site_id).xpos
        err = np.asarray(target_xyz) - site_pos
        if np.linalg.norm(err) < _TOLERANCE:
            break
        mujoco.mj_jacSite(model, scratch, jacp, None, site_id)
        J = jacp[:, dof_idx]  # (3, 6)
        JJt = J @ J.T + (_DAMPING**2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err)
        for k, qi in enumerate(qpos_idx):
            new_val = scratch.qpos[qi] + _STEP_SIZE * dq[k]
            scratch.qpos[qi] = np.clip(new_val, joint_ranges[k, 0], joint_ranges[k, 1])
        mujoco.mj_forward(model, scratch)

    return np.array([scratch.qpos[qi] for qi in qpos_idx])
