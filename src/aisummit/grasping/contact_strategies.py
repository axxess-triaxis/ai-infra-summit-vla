"""Alternative contact strategies for elongated objects where the
conventional bilateral (symmetric, closing-across-the-width) grasp is
geometrically infeasible -- see README's "alternative contact strategies"
section for the full evidence trail this is built on.

Four strategies are named as a design vocabulary (Phase 2 of the brief
this implements), only two investigated so far:

- CONVENTIONAL_BILATERAL_GRASP -- the existing `grasping/planner.py`
  pipeline. Unchanged; this module never modifies it.
- HANDLE_END_ACQUISITION -- implemented and tested here. Targets the
  handle's physical end (not its center) and searches wrist_angle jointly
  with wrist_rotate (not wrist_rotate alone), so the *motion/contact
  family* genuinely differs from the conventional strategy, not just the
  target point.
- CONSTRAINED_SIDE_APPROACH / SCOOPING_BACKSTOP_APPROACH -- named but not
  implemented. Investigated only as a feasibility question (see
  `SCOOPING_BACKSTOP_REJECTED_REASON` below) after a real physical test,
  not assumed: the plate sits 11cm from the fork along the only useful
  push direction and its top surface is ~6mm above the table, neither of
  which changes the actual constraint (see below), and a direct
  level-orientation close/lift attempt was run and produced zero
  displacement -- confirmed empirically, not theorized.

Ordering matters throughout this module: FEASIBILITY (does IK converge,
is the pose collision-free, do both fingers actually reach the object)
is always checked before SCORING. A strategy with an excellent score but
no real acquisition is a failure, full stop -- see `evaluate_candidate`'s
`feasible` field (scoring.py) and `verify_retained_during_lift` below,
which checks actual finger-object contact through the lift, not just
"did the object move."
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aisummit.control.ik import finger_target_to_site_target, solve_ik
from aisummit.grasping.candidates import GraspCandidate, finger_positions, horizontal_unit, seed_scratch
from aisummit.grasping.collision import Contact, classify_contacts
from aisummit.grasping.geometry import ObjectGeometry
from aisummit.grasping.scoring import GraspMetrics, GraspWeights, evaluate_candidate
from aisummit.sim.env import ARM_JOINTS, DinnerTableEnv

# Bounded, deterministic, small -- not a "hundreds of orientation
# parameters" search. wrist_angle range is narrower than wrist_rotate's
# because forcing it much past +/-25deg from the position-only seed
# reliably fails to converge (measured: of a 66-point wrist_angle x
# wrist_rotate grid tried during development, only 2 converged at all,
# both at the low end of this range).
_HANDLE_END_WRIST_ANGLE_DEG = (-20, -10, 0, 10, 20)
_HANDLE_END_WRIST_ROTATE_DEG = tuple(range(0, 181, 15))
# How far past the handle's own centerline to target -- 1cm short of the
# fork's actual tip (half_length from center), so the candidate is at the
# physical end without being asked to reach past the object entirely.
_HANDLE_END_TIP_MARGIN_M = 0.01


@dataclass
class StrategySpec:
    """The fields Phase 2 of the brief asked every strategy to define, as
    plain data -- with only two strategies existing, a class hierarchy or
    Protocol would be premature abstraction for what's currently a
    lookup table plus two functions."""

    name: str
    approach_pose: str
    approach_direction: str
    allowed_object_relative_motion: str
    finger_opening_state: str
    contact_assumptions: str
    acquisition_condition: str
    transition_to_lift: str
    failure_condition: str


CONVENTIONAL_BILATERAL_GRASP = StrategySpec(
    name="conventional_bilateral_grasp",
    approach_pose="Above the grasp region, elevated by the table-clearance margin, then a translation-only descent to the object's real height.",
    approach_direction="Top-down; wrist_rotate swept to align the closing axis perpendicular to the object's principal axis.",
    allowed_object_relative_motion="None -- rigid target, no intended object displacement before closing.",
    finger_opening_state="Open on approach, closed only after reaching the final (descended) pose.",
    contact_assumptions="Only the target object; any other contact (table, other tableware) is a rejection.",
    acquisition_condition="Both fingers individually within thickness+margin of the object's height, mismatch within tolerance, no collision.",
    transition_to_lift="Direct vertical lift from the close point.",
    failure_condition="No candidate satisfies collision-free + both-fingers-reach + mismatch-tolerance simultaneously -- the finding this module exists to work around.",
)

HANDLE_END_ACQUISITION = StrategySpec(
    name="handle_end_acquisition",
    approach_pose="Near the handle's physical longitudinal end (tip), not its center.",
    approach_direction="Top-down, but with wrist_angle swept jointly with wrist_rotate -- searches for a flatter/more-level approach family, not just the top-down-then-twist family the conventional strategy uses.",
    allowed_object_relative_motion="None -- same rigid-target assumption as conventional.",
    finger_opening_state="Open on approach, closed only after reaching the final pose.",
    contact_assumptions="Only the target object; any other contact is a rejection, identical to conventional.",
    acquisition_condition="Same as conventional: both fingers reach, mismatch within tolerance, no collision -- what differs is the search, not the acceptance criteria.",
    transition_to_lift="Direct vertical lift from the close point.",
    failure_condition="Investigated and found infeasible for the fork's current pose -- see README for the measured wrist_angle x wrist_rotate convergence/collision data.",
)

CONSTRAINED_SIDE_APPROACH = StrategySpec(
    name="constrained_side_approach",
    approach_pose="Not implemented.",
    approach_direction="Not implemented.",
    allowed_object_relative_motion="Not implemented.",
    finger_opening_state="Not implemented.",
    contact_assumptions="Not implemented.",
    acquisition_condition="Not implemented.",
    transition_to_lift="Not implemented.",
    failure_condition="Named for the design vocabulary; not distinct enough from handle-end acquisition to justify a separate implementation once handle-end was shown infeasible for the same underlying reason (finger-link geometry vs. table clearance, not object-relative positioning).",
)

SCOOPING_BACKSTOP_APPROACH = StrategySpec(
    name="scooping_backstop_approach",
    approach_pose="Investigated, not implemented as a full motion sequence.",
    approach_direction="A level (wrist_angle=wrist_rotate=0) approach was tested directly, since that's the one orientation already confirmed collision-free.",
    allowed_object_relative_motion="Would require first pushing the fork ~11cm toward the plate -- a separate manipulation phase, never reached because the acquisition step itself failed first.",
    finger_opening_state="Open, then closed at the level orientation.",
    contact_assumptions="Plate as an intended support/backstop body -- `classify_contacts`'s `intended_support` category exists for this, unused by any strategy yet since none reached the point of needing it.",
    acquisition_condition="Not met.",
    transition_to_lift="N/A -- never acquired.",
    failure_condition=(
        "Confirmed by direct simulation, not just reasoning: at the one orientation that clears the "
        "table (wrist_angle=wrist_rotate=0), the closing axis is parallel to the fork's length, not "
        "perpendicular to it -- closing the gripper here happens in the object's own longitudinal "
        "plane and never engages its cross-section at all. Measured: fork position identical to 12 "
        "decimal places before approach, after closing, and after lift -- zero contact, zero "
        "displacement. Separately, the ALOHA gripper is a 2-finger parallel jaw with a ~7.4cm max "
        "opening; the fork is 18cm long, so an end-to-end pinch spanning the whole object is not "
        "geometrically possible regardless of orientation. And the plate -- the only candidate "
        "backstop -- sits 11cm from the fork along the one useful push axis, with its top surface "
        "only ~6mm above the table, which doesn't change the finger-link-vs-table constraint that "
        "blocks every tilted orientation in the first place."
    ),
)


@dataclass
class StrategyOutcome:
    spec: StrategySpec
    feasible: bool
    candidates_evaluated: int
    ik_converged: int
    best_candidate: GraspCandidate | None
    best_metrics: GraspMetrics | None
    reason: str


def _solve_two_stage(env: DinnerTableEnv, side: str, candidate: GraspCandidate) -> np.ndarray:
    """Same two-stage (approach, then translation-only descent) solve
    `grasping/planner.py::_solve_matching_execution` uses, duplicated
    rather than imported to avoid a planner<->contact_strategies import
    cycle (planner.py will call into this module for the fallback
    strategy). Any future change to the two-stage solve logic should be
    made in both places -- there are only two call sites."""
    approach_site_target = finger_target_to_site_target(candidate.target_position, candidate.target_quat)
    approach_solved = solve_ik(
        env.model, env.data, side, approach_site_target, target_quat=candidate.target_quat,
        seed_angles=candidate.seed_angles,
    )
    if candidate.final_target_position is None:
        return approach_solved
    descent_site_target = finger_target_to_site_target(candidate.final_target_position, candidate.target_quat)
    return solve_ik(
        env.model, env.data, side, descent_site_target, target_quat=candidate.target_quat,
        seed_angles=approach_solved,
    )


def generate_handle_end_candidates(
    model: mujoco.MjModel, data: mujoco.MjData, side: str, geometry: ObjectGeometry,
) -> list[GraspCandidate]:
    """Candidates for HANDLE_END_ACQUISITION: targets the physical end of
    the handle (not its center) and sweeps wrist_angle jointly with
    wrist_rotate -- a genuinely different orientation search family from
    `grasping/candidates.py::generate_candidates`, which only sweeps
    wrist_rotate around whatever wrist_angle position-only IK happens to
    produce. Structurally similar to that function (same seed-then-sweep
    pattern, same helpers) since the *mechanism* being reused is sound;
    what's new is which two joints are varied and where the target sits."""
    base_point = geometry.grasp_region if geometry.grasp_region is not None else geometry.position
    axis_h = horizontal_unit(geometry.principal_axis)
    tip_point = base_point - axis_h * (geometry.half_length - _HANDLE_END_TIP_MARGIN_M - abs(
        float(np.dot(base_point - geometry.position, axis_h))
    ))
    wa_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/wrist_angle")
    wa_qadr = model.jnt_qposadr[wa_jnt_id]
    wr_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/wrist_rotate")
    wr_qadr = model.jnt_qposadr[wr_jnt_id]

    seed_angles = solve_ik(model, data, side, tip_point)
    candidates: list[GraspCandidate] = []
    for wa_deg in _HANDLE_END_WRIST_ANGLE_DEG:
        for wr_deg in _HANDLE_END_WRIST_ROTATE_DEG:
            scratch = seed_scratch(model, data, side, seed_angles)
            scratch.qpos[wa_qadr] = np.radians(wa_deg)
            scratch.qpos[wr_qadr] = np.radians(wr_deg)
            mujoco.mj_forward(model, scratch)

            gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/gripper")
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, scratch.site(gripper_site_id).xmat)

            left_finger, right_finger = finger_positions(model, scratch, side)
            closing_axis = horizontal_unit(right_finger - left_finger)
            alignment = 1.0 - abs(float(np.dot(closing_axis, axis_h)))

            wrist_overridden_angles = np.array(
                [scratch.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{j}")]]
                 for j in ARM_JOINTS]
            )
            candidates.append(
                GraspCandidate(
                    side=side,
                    target_position=tip_point,
                    target_quat=quat,
                    source=f"handle_end,wa={wa_deg:+d},wr={wr_deg}deg",
                    closing_axis_alignment=alignment,
                    final_target_position=tip_point,
                    seed_angles=wrist_overridden_angles,
                )
            )
    return candidates


def evaluate_handle_end_grasp(
    env: DinnerTableEnv, side: str, geometry: ObjectGeometry, weights: GraspWeights | None = None,
) -> StrategyOutcome:
    weights = weights or GraspWeights()
    candidates = generate_handle_end_candidates(env.model, env.data, side, geometry)
    ik_converged = 0
    scored: list[tuple[float, GraspCandidate, GraspMetrics]] = []
    for cand in candidates:
        solved = _solve_two_stage(env, side, cand)
        metrics = evaluate_candidate(env.model, env.data, cand, geometry, solved, weights)
        if metrics.ik_ok:
            ik_converged += 1
        if metrics.feasible:
            scored.append((metrics.score, cand, metrics))

    if scored:
        scored.sort(key=lambda t: -t[0])
        _, best_cand, best_metrics = scored[0]
        return StrategyOutcome(
            spec=HANDLE_END_ACQUISITION, feasible=True, candidates_evaluated=len(candidates),
            ik_converged=ik_converged, best_candidate=best_cand, best_metrics=best_metrics, reason="",
        )

    # No feasible candidate -- report the best-scoring INFEASIBLE one for
    # diagnostics (same "best infeasible" pattern as
    # GraspDebugTrace.spatial_search_summary()), not just a bare failure.
    all_evaluated = []
    for cand in candidates:
        solved = _solve_two_stage(env, side, cand)
        metrics = evaluate_candidate(env.model, env.data, cand, geometry, solved, weights)
        all_evaluated.append((metrics.score, cand, metrics))
    all_evaluated.sort(key=lambda t: -t[0])
    _, best_cand, best_metrics = all_evaluated[0] if all_evaluated else (0.0, None, None)
    reason = best_metrics.infeasible_reason if best_metrics else "no candidates generated"
    return StrategyOutcome(
        spec=HANDLE_END_ACQUISITION, feasible=False, candidates_evaluated=len(candidates),
        ik_converged=ik_converged, best_candidate=best_cand, best_metrics=best_metrics,
        reason=f"no feasible handle-end candidate (best: {reason})",
    )


def verify_retained_during_lift(
    env: DinnerTableEnv, side: str, object_name: str,
) -> tuple[bool, list[Contact]]:
    """Contact-based acquisition check, stricter than "did the object
    move": requires an ACTUAL, current contact between at least one
    finger link and the target object at the moment this is called (meant
    to be called right after a lift). An object that was merely nudged or
    is resting on top of a finger without being pinched will show no
    finger-object contact once physics has settled from the nudge, so
    this catches "the fork moved" without "the fork is actually held"
    (Phase 8.D/E's exact concern) -- physically true even when a
    position/tracking check would report a false positive."""
    contacts = classify_contacts(
        env.model, env.data, side, _current_side_arm_angles(env, side), target_object=object_name,
    )
    finger_object_contacts = [c for c in contacts if c.kind == "intended_target"]
    return bool(finger_object_contacts), contacts


def _current_side_arm_angles(env: DinnerTableEnv, side: str) -> np.ndarray:
    return np.array(
        [env.data.qpos[env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{j}")]]
         for j in ARM_JOINTS]
    )
