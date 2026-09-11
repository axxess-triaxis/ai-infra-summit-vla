"""Top-level grasp orchestration: geometry -> candidates -> score -> select
-> IK -> collision check -> execute -> verify -> recover.

`grasp_object` is the single entrypoint the rest of the system should call
for a "pick" action. Radially symmetric objects take the exact old
position-only `pick()` path, completely unchanged -- this function only
introduces new behavior for elongated objects, per the hackathon brief's
"preserve all currently working functionality."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aisummit.control.ik import finger_target_to_site_target, solve_ik
from aisummit.control.primitives import (
    GRIPPER_OPEN,
    hold,
    move_to,
    pick,
    pick_oriented,
    set_gripper,
)
from aisummit.grasping.candidates import GraspCandidate, generate_candidates
from aisummit.grasping.debug import AttemptTrace, CandidateTrace, GraspDebugTrace
from aisummit.grasping.geometry import ObjectGeometry, estimate_object_geometry
from aisummit.grasping.scoring import GraspWeights, evaluate_candidate, stability_score
from aisummit.sim.env import DinnerTableEnv, Observation

_STABILIZATION_STABILITY_THRESHOLD = 0.3
_LIFT_VERIFICATION_MIN_HEIGHT = 0.03
_LIFT_VERIFICATION_MAX_XY_DEVIATION = 0.06


@dataclass
class GraspResult:
    success: bool
    side: str
    object_name: str
    trace: GraspDebugTrace
    final_ctrl: np.ndarray
    achieved_quat: np.ndarray | None = None


def _verify_lifted(env: DinnerTableEnv, obs: Observation, object_name: str, side: str, start_height: float) -> bool:
    obj_pos = obs.object_positions[object_name]
    # The true grasp point is the finger midpoint, not the {side}/gripper
    # site `obs.gripper_positions` reports (~1.4cm apart -- see ik.py).
    # The 6cm tolerance here made this forgiving enough not to matter
    # before, but computing it correctly costs nothing.
    finger_mid = (env.site_xpos(f"{side}/left_finger") + env.site_xpos(f"{side}/right_finger")) / 2
    lifted = obj_pos[2] > start_height + _LIFT_VERIFICATION_MIN_HEIGHT
    tracking = np.linalg.norm(obj_pos[:2] - finger_mid[:2]) < _LIFT_VERIFICATION_MAX_XY_DEVIATION
    return bool(lifted and tracking)


def _solve_matching_execution(env: DinnerTableEnv, side: str, candidate: GraspCandidate) -> np.ndarray:
    """Solves IK exactly the way `pick_oriented` will actually execute --
    a single direct solve straight to the grasp pose, warm-started from the
    candidate's own seed configuration, no intermediate waypoint.

    Two things had to be fixed to make this trustworthy: an earlier version
    chained through a high-approach waypoint first (mirroring an earlier
    `pick_oriented`), which measurably hurt convergence (a kinematic elbow
    flip) -- `pick_oriented` moved to a single-stage approach for exactly
    that reason. Separately, solving cold-start (from the arm's resting
    pose) failed to converge for far/twisted candidates that a warm-started
    solve reaches easily -- found via a real false positive: a candidate
    that looked collision-free and well-aligned on paper was actually
    sitting nowhere near its intended target, because nothing had checked
    the *solved* pose's position residual, only whether it collided with
    anything. Keeping both the waypoint structure and the seed in lockstep
    with `pick_oriented` is what makes scoring/collision-checking
    predictive of what execution actually does.

    `candidate.target_position` is the true finger-midpoint grasp point --
    converted to the `{side}/gripper` site target `solve_ik` needs via
    `finger_target_to_site_target`."""
    site_target = finger_target_to_site_target(candidate.target_position, candidate.target_quat)
    return solve_ik(
        env.model, env.data, side, site_target, target_quat=candidate.target_quat,
        seed_angles=candidate.seed_angles,
    )


def _horizontal_unit(vec: np.ndarray) -> np.ndarray:
    flat = vec.copy()
    flat[2] = 0.0
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])


def stabilize_with_other_arm(
    env: DinnerTableEnv, ctrl: np.ndarray, stabilizer_side: str, geometry: ObjectGeometry, touch_height: float = 0.03
) -> np.ndarray:
    """Rests the stabilizer arm's closed gripper near the far end of the
    object (away from the grasp region) to discourage it sliding away while
    the other arm grasps it. Position-only -- a light touch needs no
    particular orientation, so this reuses the unmodified `move_to`."""
    if geometry.grasp_region is not None:
        handle_dir = _horizontal_unit(geometry.grasp_region - geometry.position)
    else:
        handle_dir = _horizontal_unit(geometry.principal_axis)
    far_end = geometry.position - handle_dir * geometry.half_length * 0.7
    ctrl = set_gripper(env, ctrl, stabilizer_side, GRIPPER_OPEN * 0.3)
    ctrl, _ = move_to(env, ctrl, stabilizer_side, far_end + np.array([0, 0, 0.08]))
    ctrl, _ = move_to(env, ctrl, stabilizer_side, far_end + np.array([0, 0, touch_height]))
    hold(env, ctrl, steps=20)
    return ctrl


def retract_arm(env: DinnerTableEnv, ctrl: np.ndarray, side: str, retreat_height: float = 0.20) -> np.ndarray:
    base_body_id_pos = env.body_xpos(f"{side}/base_link")
    ctrl = set_gripper(env, ctrl, side, GRIPPER_OPEN)
    ctrl, _ = move_to(env, ctrl, side, base_body_id_pos + np.array([0.15, 0, retreat_height]))
    return ctrl


def grasp_object(
    env: DinnerTableEnv,
    ctrl: np.ndarray,
    side: str,
    object_name: str,
    weights: GraspWeights | None = None,
    debug: GraspDebugTrace | None = None,
    allow_stabilization: bool = False,
    stabilizer_side: str | None = None,
) -> GraspResult:
    weights = weights or GraspWeights()
    trace = debug if debug is not None else GraspDebugTrace(object_name=object_name)
    geometry = estimate_object_geometry(env.model, env.data, object_name)

    if not geometry.is_elongated:
        trace.geometry = geometry
        start_height = float(env.body_xpos(object_name)[2])
        trace.log("radially symmetric -- unmodified position-only pick()")
        ctrl, obs = pick(env, ctrl, side, object_name)
        success = _verify_lifted(env, obs, object_name, side, start_height)
        return GraspResult(success, side, object_name, trace, ctrl)

    # Authored object heights aren't exactly resting-contact height (found
    # empirically: the fork/knife/plate start ~1.6cm above the table).
    # Settling here (not in env.reset(), which both grasp paths share --
    # doing it there shifted the cup regression test's numbers enough to
    # expose a separate, pre-existing fragility in place()'s release step)
    # means the geometry candidates get generated against matches where the
    # object will actually be by the time the arm gets there.
    env.settle()
    geometry = estimate_object_geometry(env.model, env.data, object_name)
    trace.geometry = geometry
    start_height = float(env.body_xpos(object_name)[2])

    if allow_stabilization and stabilizer_side and stability_score(geometry) < _STABILIZATION_STABILITY_THRESHOLD:
        trace.log(f"low predicted stability ({stability_score(geometry):.2f}) -- stabilizing with {stabilizer_side} arm")
        ctrl = stabilize_with_other_arm(env, ctrl, stabilizer_side, geometry)

    candidates = generate_candidates(env.model, env.data, side, geometry)
    scored: list[tuple[float, GraspCandidate, np.ndarray]] = []
    for cand in candidates:
        solved_angles = _solve_matching_execution(env, side, cand)
        metrics = evaluate_candidate(env.model, env.data, cand, geometry, solved_angles, weights)
        if not metrics.collision_valid:
            trace.record_candidate(CandidateTrace(cand.source, metrics, False, metrics.collision_reason))
            continue
        # workspace_margin is an admitted approximation (see scoring.py's
        # _NOMINAL_MAX_REACH_M docstring) -- it stays a continuous, weighted
        # scoring signal but is NOT a hard reject gate. It was rejecting a
        # candidate that ik_ok and collision_valid both confirmed as a real,
        # reachable, collision-free solution (right arm reaching across for
        # the fork, 0.69m from its base -- comfortably beyond the 0.5m
        # guess). ik_ok and collision_valid are exact simulator facts;
        # workspace_margin is not, so it shouldn't get veto power over them.
        if not metrics.ik_ok:
            trace.record_candidate(CandidateTrace(cand.source, metrics, False, "IK did not converge"))
            continue
        trace.record_candidate(CandidateTrace(cand.source, metrics, True))
        scored.append((metrics.score, cand, solved_angles))

    if not scored:
        trace.log("no valid candidates survived filtering/scoring")
        if allow_stabilization and stabilizer_side:
            ctrl = retract_arm(env, ctrl, stabilizer_side)
        return GraspResult(False, side, object_name, trace, ctrl)

    scored.sort(key=lambda t: -t[0])
    if trace.selected is None:
        best_source = scored[0][1].source
        trace.selected = next(c for c in trace.candidates if c.source == best_source)

    final_obs = None
    for _, cand, _ in scored:
        ctrl, final_obs = pick_oriented(
            env, ctrl, side, cand.target_position, cand.target_quat, seed_angles=cand.seed_angles
        )
        verified = _verify_lifted(env, final_obs, object_name, side, start_height)
        trace.record_attempt(AttemptTrace(cand.source, verified))
        if verified:
            if allow_stabilization and stabilizer_side:
                ctrl = retract_arm(env, ctrl, stabilizer_side)
            return GraspResult(True, side, object_name, trace, ctrl, achieved_quat=cand.target_quat)

        # Release and retreat before the next candidate -- never re-issue an
        # identical failed grasp blindly.
        ctrl = set_gripper(env, ctrl, side, GRIPPER_OPEN)
        hold(env, ctrl, steps=20)
        ctrl, _ = move_to(env, ctrl, side, cand.target_position + np.array([0, 0, 0.15]))

    trace.log("all candidates attempted, none verified as grasped")
    if allow_stabilization and stabilizer_side:
        ctrl = retract_arm(env, ctrl, stabilizer_side)
    return GraspResult(False, side, object_name, trace, ctrl)
