"""Damped least-squares numerical IK for one ALOHA arm.

Solves against the `{side}/gripper` site's 6 arm joints. Runs on a scratch
copy of MjData so it never disturbs the live simulation while it iterates.

`target_quat` is optional and additive: omit it (the default) and this is
the exact same position-only 3-constraint solve it always was -- every
existing caller (`control/primitives.py::move_to`) is unaffected. Pass it
and the solve becomes a fully-determined 6-constraint (3 position + 3
orientation) problem, which is what `grasping/` uses for elongated objects.
This was extended rather than replaced per the hackathon brief's "don't
rewrite the IK unless necessary" -- the Jacobian call already computed
underneath (`mj_jacSite`) exposes the rotational Jacobian for free via its
second output argument, which the position-only path simply passed `None`
for.
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


def _orientation_error(scratch: mujoco.MjData, site_id: int, target_quat: np.ndarray) -> np.ndarray:
    """3-vector tangent-space rotation from the site's current orientation to `target_quat`."""
    cur_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cur_quat, scratch.site(site_id).xmat)
    err = np.zeros(3)
    mujoco.mju_subQuat(err, np.asarray(target_quat), cur_quat)
    return err


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    target_xyz: np.ndarray,
    target_quat: np.ndarray | None = None,
    max_iters: int = _MAX_ITERS,
) -> np.ndarray:
    """Returns the 6 target joint angles for `side`'s arm that bring its
    gripper site closest to `target_xyz` (and, if given, `target_quat`, a
    world-frame quaternion in wxyz order). Does not mutate `data`."""
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
    jacr = np.zeros((3, model.nv)) if target_quat is not None else None
    ndim = 6 if target_quat is not None else 3
    damping_eye = (_DAMPING**2) * np.eye(ndim)

    for _ in range(max_iters):
        pos_err = np.asarray(target_xyz) - scratch.site(site_id).xpos
        if target_quat is None:
            err = pos_err
        else:
            err = np.concatenate([pos_err, _orientation_error(scratch, site_id, target_quat)])
        if np.linalg.norm(err) < _TOLERANCE:
            break

        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = jacp[:, dof_idx] if target_quat is None else np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        JJt = J @ J.T + damping_eye
        dq = J.T @ np.linalg.solve(JJt, err)
        for k, qi in enumerate(qpos_idx):
            new_val = scratch.qpos[qi] + _STEP_SIZE * dq[k]
            scratch.qpos[qi] = np.clip(new_val, joint_ranges[k, 0], joint_ranges[k, 1])
        mujoco.mj_forward(model, scratch)

    return np.array([scratch.qpos[qi] for qi in qpos_idx])


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(world position, world quaternion wxyz) of a named site, read via forward kinematics."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site(site_id).xmat)
    return data.site(site_id).xpos.copy(), quat
