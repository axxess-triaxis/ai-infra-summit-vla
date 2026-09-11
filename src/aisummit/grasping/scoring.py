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

# Extra allowance beyond the object's own full thickness before a
# finger-height mismatch is treated as fully disqualifying. Small and
# object-agnostic on purpose -- it represents aiming/compliance slop, not
# a per-object tuning knob (object size itself already comes from
# `geometry.thickness`, which is what makes the tolerance object-aware).
_FINGER_HEIGHT_MARGIN_M = 0.005


@dataclass
class GraspWeights:
    ik_success: float = 3.0
    orientation_alignment: float = 2.0
    finger_height_symmetry: float = 2.0
    config_distance: float = 0.5
    grasp_region_confidence: float = 0.5
    workspace_margin: float = 1.0
    stability: float = 1.0

    def total(self) -> float:
        return (
            self.ik_success
            + self.orientation_alignment
            + self.finger_height_symmetry
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
    finger_left_z: float
    finger_right_z: float
    finger_height_mismatch: float
    finger_height_symmetry_score: float
    left_finger_reaches_object: bool
    right_finger_reaches_object: bool
    feasible: bool
    infeasible_reason: str
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


def _finger_reaches_object(finger_z: float, geometry: ObjectGeometry) -> bool:
    """Whether a single finger's height is anywhere near the object's own
    vertical extent -- distinct from finger_height_mismatch (which only
    compares the two fingers to EACH OTHER, so a candidate with both
    fingers equally far above the object would score perfectly symmetric
    while still touching nothing). Found necessary by direct measurement:
    the fork's best-aligned candidate had left_finger at 20.8mm and
    right_finger at 61.1mm while the fork itself spans roughly
    [-1.9mm, 6.1mm] -- neither finger individually reaches, which a
    symmetry-only check cannot see."""
    obj_z = geometry.position[2]
    return bool(abs(finger_z - obj_z) <= geometry.thickness + _FINGER_HEIGHT_MARGIN_M)


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
    left_finger_pos = scratch.site(left_finger_id).xpos
    right_finger_pos = scratch.site(right_finger_id).xpos
    achieved_pos = (left_finger_pos + right_finger_pos) / 2

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{candidate.side}/gripper")
    achieved_quat = np.zeros(4)
    mujoco.mju_mat2Quat(achieved_quat, scratch.site(site_id).xmat)

    # When a candidate has a translation-only final descent
    # (final_target_position), `solved_angles` is the POST-descent
    # configuration (see planner.py's _solve_matching_execution) -- so the
    # achieved pose must be compared against where the arm actually closes,
    # not the higher approach point it passed through on the way there.
    intended_position = (
        candidate.final_target_position if candidate.final_target_position is not None
        else candidate.target_position
    )
    pos_err = float(np.linalg.norm(achieved_pos - intended_position))
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
    reach_dist = float(np.linalg.norm(intended_position - data.xpos[base_body_id]))
    workspace_margin = float(np.clip(1.0 - reach_dist / _NOMINAL_MAX_REACH_M, 0.0, 1.0))

    # A wrist angle can score well on horizontal closing-axis alignment
    # while still tilting the gripper enough that the two fingers sit at
    # noticeably different heights -- found via direct physical inspection
    # of the fork: the best-*aligned* candidate left the fingers ~1cm apart
    # in height, well above the fork's 8mm thickness, so it closed above/
    # below the object rather than around it. `finger_height_mismatch` is
    # measured directly from the same solved-angle forward kinematics as
    # every other metric here, never assumed. The tolerance is object-aware
    # (scales with `geometry.thickness`) rather than a fixed constant, so a
    # thick object gets a lenient allowance and a thin one a strict one --
    # same clipped-linear shape already used for config_distance_score and
    # workspace_margin above, not a new scoring idiom.
    finger_height_mismatch = float(abs(left_finger_pos[2] - right_finger_pos[2]))
    allowed_height_mismatch = 2.0 * geometry.thickness + _FINGER_HEIGHT_MARGIN_M
    finger_height_symmetry_score = float(np.clip(1.0 - finger_height_mismatch / allowed_height_mismatch, 0.0, 1.0))
    left_reaches = _finger_reaches_object(float(left_finger_pos[2]), geometry)
    right_reaches = _finger_reaches_object(float(right_finger_pos[2]), geometry)

    # FEASIBILITY is a hard classification, separate from the weighted
    # SCORE below -- a candidate that fails any of these is physically
    # incapable of the grasp, and no amount of alignment should buy it a
    # win (requirement: "do not allow a very high alignment score to
    # compensate for a physically impossible bilateral grasp"). Checked in
    # a fixed order so `infeasible_reason` names the first real problem,
    # not just "infeasible".
    if not ik_ok:
        feasible, infeasible_reason = False, "IK did not converge"
    elif not collision_valid:
        feasible, infeasible_reason = False, collision_reason
    elif not left_reaches and not right_reaches:
        feasible, infeasible_reason = False, "neither finger reaches the object"
    elif not left_reaches:
        feasible, infeasible_reason = False, "left finger cannot reach the object"
    elif not right_reaches:
        feasible, infeasible_reason = False, "right finger cannot reach the object"
    elif finger_height_symmetry_score <= 0.0:
        feasible, infeasible_reason = False, "finger-height mismatch exceeds object-aware tolerance"
    else:
        feasible, infeasible_reason = True, ""

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
        finger_left_z=float(left_finger_pos[2]),
        finger_right_z=float(right_finger_pos[2]),
        finger_height_mismatch=finger_height_mismatch,
        finger_height_symmetry_score=finger_height_symmetry_score,
        left_finger_reaches_object=left_reaches,
        right_finger_reaches_object=right_reaches,
        feasible=feasible,
        infeasible_reason=infeasible_reason,
    )

    total = weights.total()
    metrics.score = (
        weights.ik_success * (1.0 if ik_ok else 0.0)
        + weights.orientation_alignment * metrics.orientation_alignment
        + weights.finger_height_symmetry * metrics.finger_height_symmetry_score
        + weights.config_distance * config_distance_score
        + weights.grasp_region_confidence * metrics.grasp_region_confidence
        + weights.workspace_margin * metrics.workspace_margin
        + weights.stability * metrics.stability
    ) / total
    return metrics
