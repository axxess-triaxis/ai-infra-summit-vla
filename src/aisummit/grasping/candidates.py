"""Orientation-aware grasp candidate generation for elongated objects.

Strategy (chosen to minimize risk to the existing IK rather than for
theoretical elegance -- see control/ik.py's docstring and the README's
"orientation-aware grasping" section for the full reasoning):

1. Reuse the EXISTING, unmodified position-only `solve_ik` to get a joint
   config that already reaches near the candidate grasp point. This is
   requirement-3-option-B in spirit ("continue to call the existing
   position IK") and gives the 6-DOF orientation solve a warm start close
   to the answer, rather than hand-deriving a target quaternion from
   scratch (which would require knowing the gripper site's local axis
   convention -- a real risk of silently getting a sign/axis wrong).
2. From that seed, sweep the arm's own `wrist_rotate` joint (which barely
   moves the gripper's position -- it's the last joint before the gripper,
   close to a spherical-wrist assumption) through several candidate angles,
   and read the ACTUAL resulting gripper orientation via forward kinematics.
   No axis convention is assumed anywhere -- we measure it.
3. Score each swept angle by how perpendicular the gripper's real
   finger-to-finger line (read directly from the model's own
   `{side}/left_finger` / `{side}/right_finger` sites) ends up to the
   object's principal axis -- that's the geometric condition for the
   fingers to close across the object's width rather than along its length.
4. Feed the resulting (position, quaternion) pair into the *extended*
   `solve_ik(..., target_quat=...)` for a final precise 6-DOF solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aisummit.control.ik import solve_ik
from aisummit.grasping.geometry import ObjectGeometry
from aisummit.sim.env import ARM_JOINTS

_POSITION_OFFSETS = (-0.015, 0.0, 0.015)  # meters, along the object's principal axis
# The authored handle site sits at the object's exact centerline height,
# right at the tabletop -- fine for *where* to grasp, but it leaves the
# gripper's body almost no vertical clearance before wider wrist angles
# drive it into the table (found empirically: several otherwise-good
# candidates were rejected for 1-5cm table penetration). Lifting the
# target a little is a grasp-planning decision, not a geometry-estimation
# one, so it lives here rather than adjusting the authored handle height.
_GRASP_HEIGHT_CLEARANCE = 0.015
# 15-degree steps rather than 30: the viable window that clears the table
# while keeping decent finger-closing alignment turned out to be fairly
# narrow (empirically ~70-90deg for this arm's geometry, found by sweeping
# 10-degree steps during debugging) -- 30-degree steps skipped over it
# entirely. Still cheap: this is deterministic FK, not an LLM call.
_WRIST_SWEEP_DEG = tuple(range(0, 181, 15))  # degrees; 180 deg-periodic for a parallel gripper


@dataclass
class GraspCandidate:
    side: str
    target_position: np.ndarray
    target_quat: np.ndarray
    source: str  # human-readable provenance, e.g. "offset=+0.015,wrist=90deg"
    closing_axis_alignment: float  # 0..1, 1 = perfectly perpendicular to the object's axis


def _finger_positions(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[np.ndarray, np.ndarray]:
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/right_finger")
    return data.site(left_id).xpos.copy(), data.site(right_id).xpos.copy()


def _seed_scratch(model: mujoco.MjModel, data: mujoco.MjData, side: str, arm_angles: np.ndarray) -> mujoco.MjData:
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    for j, joint_name in enumerate(ARM_JOINTS):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{joint_name}")
        scratch.qpos[model.jnt_qposadr[jnt_id]] = arm_angles[j]
    return scratch


def _horizontal_unit(vec: np.ndarray) -> np.ndarray:
    flat = vec.copy()
    flat[2] = 0.0
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])


def generate_candidates(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    geometry: ObjectGeometry,
    position_offsets: tuple[float, ...] = _POSITION_OFFSETS,
    wrist_sweep_deg: tuple[float, ...] = _WRIST_SWEEP_DEG,
) -> list[GraspCandidate]:
    base_point = geometry.grasp_region if geometry.grasp_region is not None else geometry.position
    base_point = base_point + np.array([0, 0, _GRASP_HEIGHT_CLEARANCE])
    axis_h = _horizontal_unit(geometry.principal_axis)
    wrist_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/wrist_rotate")
    wrist_qadr = model.jnt_qposadr[wrist_jnt_id]

    candidates: list[GraspCandidate] = []
    for offset in position_offsets:
        target_pos = base_point + offset * axis_h
        seed_angles = solve_ik(model, data, side, target_pos)  # unmodified position-only IK, reused as-is

        for wrist_deg in wrist_sweep_deg:
            scratch = _seed_scratch(model, data, side, seed_angles)
            scratch.qpos[wrist_qadr] = np.radians(wrist_deg)
            mujoco.mj_forward(model, scratch)

            gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/gripper")
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, scratch.site(gripper_site_id).xmat)

            left_finger, right_finger = _finger_positions(model, scratch, side)
            closing_axis = _horizontal_unit(right_finger - left_finger)
            alignment = 1.0 - abs(float(np.dot(closing_axis, axis_h)))

            candidates.append(
                GraspCandidate(
                    side=side,
                    target_position=target_pos,
                    target_quat=quat,
                    source=f"offset={offset:+.3f},wrist={wrist_deg}deg",
                    closing_axis_alignment=alignment,
                )
            )
    return candidates
