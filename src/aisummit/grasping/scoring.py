"""Grasp candidate scoring -- combines several cheap, deterministic signals
into one number so the planner can rank candidates instead of trusting a
single predicted grasp (requirement: "do not rely on one predicted grasp").

Every weight lives in `GraspWeights` (a dataclass, not scattered literals)
so tuning the trade-offs is one object, not a grep-and-edit across files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from aisummit.control.primitives import GRIPPER_OPEN
from aisummit.grasping.candidates import GraspCandidate
from aisummit.grasping.collision import check_collision
from aisummit.grasping.geometry import ObjectGeometry
from aisummit.sim.env import ARM_JOINTS

# Approximate, not a verified robot spec -- see scoring.py docstring on
# workspace_margin. Good enough to rank candidates relative to each other.
_NOMINAL_MAX_REACH_M = 0.5

_IK_POSITION_TOLERANCE = 0.01
_IK_ORIENTATION_TOLERANCE = 0.15  # radians (~8.6deg)


@dataclass
class GraspWeights:
    ik_success: float = 3.0
    orientation_alignment: float = 2.0
    config_distance: float = 0.5
    grasp_region_confidence: float = 0.5
    workspace_margin: float = 1.0
    stability: float = 1.0

    def total(self) -> float:
        return (
            self.ik_success
            + self.orientation_alignment
            + self.config_distance
            + self.grasp_region_confidence
            + self.workspace_margin
            + self.stability
        )


@dataclass
class GraspMetrics:
    ik_position_error: float
    ik_orientation_error: float
    ik_ok: bool
    collision_valid: bool
    collision_reason: str
    config_distance: float
    workspace_margin: float
    stability: float
    orientation_alignment: float
    grasp_region_confidence: float
    score: float = 0.0


def _current_arm_angles(data, model, side: str) -> np.ndarray:
    return np.array(
        [data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{j}")]]
         for j in ARM_JOINTS]
    )


def stability_score(geometry: ObjectGeometry) -> float:
    """1.0 = object width sits comfortably inside the gripper's max opening;
    0.0 = physically too wide to close around."""
    width = 2 * geometry.radius
    if width >= GRIPPER_OPEN * 2:  # each finger travels GRIPPER_OPEN from center, roughly
        return 0.0
    return float(np.clip(1.0 - width / (GRIPPER_OPEN * 2), 0.0, 1.0))


def evaluate_candidate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    candidate: GraspCandidate,
    geometry: ObjectGeometry,
    solved_angles: np.ndarray,
    weights: GraspWeights,
) -> GraspMetrics:
    from aisummit.control.ik import solve_ik  # local import avoids a cycle at module load

    # Re-derive the achieved pose from the solved joint angles via forward
    # kinematics, so the metrics reflect what the arm will actually do.
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    for j, joint_name in enumerate(ARM_JOINTS):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{candidate.side}/{joint_name}")
        scratch.qpos[model.jnt_qposadr[jnt_id]] = solved_angles[j]
    mujoco.mj_forward(model, scratch)

    # Compare against the actual finger-closing midpoint, not the
    # `{side}/gripper` site -- they're ~1.4cm apart (see ik.py's
    # GRIPPER_SITE_TO_FINGER_MIDPOINT_OFFSET docstring), which is what was
    # silently sinking every fork/knife grasp attempt before this was found.
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{candidate.side}/left_finger")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{candidate.side}/right_finger")
    achieved_pos = (scratch.site(left_finger_id).xpos + scratch.site(right_finger_id).xpos) / 2

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{candidate.side}/gripper")
    achieved_quat = np.zeros(4)
    mujoco.mju_mat2Quat(achieved_quat, scratch.site(site_id).xmat)

    pos_err = float(np.linalg.norm(achieved_pos - candidate.target_position))
    quat_err_vec = np.zeros(3)
    mujoco.mju_subQuat(quat_err_vec, candidate.target_quat, achieved_quat)
    orient_err = float(np.linalg.norm(quat_err_vec))
    ik_ok = pos_err < _IK_POSITION_TOLERANCE and orient_err < _IK_ORIENTATION_TOLERANCE

    collision_valid, collision_reason = check_collision(
        model, data, candidate.side, solved_angles, target_object=geometry.name
    )

    current_angles = _current_arm_angles(data, model, candidate.side)
    config_distance = float(np.linalg.norm(solved_angles - current_angles))
    config_distance_score = float(np.clip(1.0 - config_distance / np.pi, 0.0, 1.0))

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{candidate.side}/base_link")
    reach_dist = float(np.linalg.norm(candidate.target_position - data.xpos[base_body_id]))
    workspace_margin = float(np.clip(1.0 - reach_dist / _NOMINAL_MAX_REACH_M, 0.0, 1.0))

    metrics = GraspMetrics(
        ik_position_error=pos_err,
        ik_orientation_error=orient_err,
        ik_ok=ik_ok,
        collision_valid=collision_valid,
        collision_reason=collision_reason,
        config_distance=config_distance,
        workspace_margin=workspace_margin,
        stability=stability_score(geometry),
        orientation_alignment=candidate.closing_axis_alignment,
        grasp_region_confidence=geometry.grasp_region_confidence,
    )

    total = weights.total()
    metrics.score = (
        weights.ik_success * (1.0 if ik_ok else 0.0)
        + weights.orientation_alignment * metrics.orientation_alignment
        + weights.config_distance * config_distance_score
        + weights.grasp_region_confidence * metrics.grasp_region_confidence
        + weights.workspace_margin * metrics.workspace_margin
        + weights.stability * metrics.stability
    ) / total
    return metrics
