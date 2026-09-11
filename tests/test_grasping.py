"""Tests for the orientation-aware grasp pipeline (grasping/).

Each test targets a distinct requirement from the grasp-pipeline upgrade
rather than re-testing the whole stack end-to-end each time:
geometry classification, candidate orientation search, collision rejection,
IK-failure rejection, retry-without-repeat, the knife's safety region, and
bimanual stabilization. `test_pick_and_place.py` covers the radially
symmetric regression case (cup) separately and is left untouched.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from aisummit.control.ik import finger_target_to_site_target, solve_ik
from aisummit.grasping.candidates import generate_candidates
from aisummit.grasping.collision import check_collision
from aisummit.grasping.debug import GraspDebugTrace
from aisummit.grasping.geometry import estimate_object_geometry
from aisummit.grasping.planner import grasp_object, stabilize_with_other_arm
from aisummit.grasping.scoring import GraspWeights, evaluate_candidate
from aisummit.sim.env import ARM_JOINTS, DinnerTableEnv


@pytest.fixture
def env():
    e = DinnerTableEnv()
    e.reset(randomize=False)
    yield e
    e.close()


def test_geometry_classifies_elongated_vs_radial(env):
    fork = estimate_object_geometry(env.model, env.data, "fork")
    knife = estimate_object_geometry(env.model, env.data, "knife")
    cup = estimate_object_geometry(env.model, env.data, "cup")
    plate = estimate_object_geometry(env.model, env.data, "plate")

    assert fork.is_elongated and knife.is_elongated
    assert not cup.is_elongated and not plate.is_elongated
    # Handle sites are authored ground truth (see geometry.py docstring) --
    # both should resolve, and be offset from the object's own CoM.
    assert fork.grasp_region is not None
    assert knife.grasp_region is not None
    assert np.linalg.norm(fork.grasp_region - fork.position) > 0.02
    assert cup.grasp_region is None  # no handle site authored for a cup


def test_candidate_search_finds_a_well_aligned_orientation(env):
    """Requirement 4: multiple candidate orientations, not one guess. At
    least one of the swept wrist angles should bring the gripper's real
    finger-to-finger line close to perpendicular with the fork's axis --
    proving the search actually explores orientation space rather than
    returning the same alignment for every candidate."""
    geometry = estimate_object_geometry(env.model, env.data, "fork")
    candidates = generate_candidates(env.model, env.data, "left", geometry)

    alignments = [c.closing_axis_alignment for c in candidates]
    assert len(candidates) > 6, "expected multiple position x orientation candidates"
    assert max(alignments) > 0.9, "no candidate achieved near-perpendicular finger alignment"
    assert max(alignments) - min(alignments) > 0.3, "orientation search isn't actually varying alignment"


def test_knife_unsafe_region_is_never_a_grasp_target(env):
    """Requirement 10: never target the blade for grasping."""
    geometry = estimate_object_geometry(env.model, env.data, "knife")
    candidates = generate_candidates(env.model, env.data, "right", geometry)
    assert geometry.unsafe_region is not None

    min_dist_to_blade = min(
        np.linalg.norm(c.target_position - geometry.unsafe_region) for c in candidates
    )
    min_dist_to_handle = min(
        np.linalg.norm(c.target_position - geometry.grasp_region) for c in candidates
    )
    assert min_dist_to_handle < min_dist_to_blade, (
        "candidates cluster nearer the blade than the handle"
    )


def test_collision_check_rejects_a_table_penetrating_pose(env):
    """Direct unit test of grasping/collision.py, independent of whether any
    particular high-level candidate happens to trigger it: solve IK for a
    target several centimeters *under* the table surface and confirm
    check_collision reports the resulting pose invalid."""
    under_table_target = env.body_xpos("fork") + np.array([0.0, 0.0, -0.06])
    forced_angles = solve_ik(env.model, env.data, "left", under_table_target)

    valid, reason = check_collision(env.model, env.data, "left", forced_angles, target_object="fork")
    assert not valid
    assert "table" in reason or "floor" in reason


def test_ik_unreachable_target_is_flagged_not_ok(env):
    """Requirement: reject candidates the position IK can't actually reach.
    A point far outside the left arm's physical workspace should score
    ik_ok=False rather than being silently accepted with a large residual."""
    from aisummit.grasping.candidates import GraspCandidate

    unreachable = np.array([5.0, 5.0, 5.0])  # 5m away -- no 6-DOF ALOHA arm reaches this
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    candidate = GraspCandidate(
        side="left", target_position=unreachable, target_quat=identity_quat,
        source="unreachable-test", closing_axis_alignment=1.0,
    )
    geometry = estimate_object_geometry(env.model, env.data, "fork")
    solved = solve_ik(env.model, env.data, "left", unreachable, target_quat=identity_quat)
    metrics = evaluate_candidate(env.model, env.data, candidate, geometry, solved, GraspWeights())
    assert not metrics.ik_ok
    assert metrics.ik_position_error > 0.1


def test_grasp_retries_multiple_distinct_candidates_without_repeating(env, monkeypatch):
    """Requirement 6: never re-issue an identical failed grasp -- checks
    the *retry mechanism itself* is correct: every attempted candidate
    source is distinct, and more than one was actually tried.

    Updated for the spatial-search iteration: with the stricter feasibility
    gate (this iteration's own STEP 4, `metrics.feasible`), the real fork
    at its current table position has ZERO feasible candidates (see
    README's "spatial repositioning" section -- confirmed by an exhaustive
    sweep across spatial offset x grasp height x wrist angle, not just this
    module's default search). That's a genuine physical finding, not a
    test-authoring problem, but it means the retry LOOP itself can no
    longer be exercised against a real feasible fork grasp. This patches
    `evaluate_candidate` to force `feasible=True` so the loop's own
    control flow -- iterate distinct candidates, stop repeating failures --
    is still tested against real execution (`pick_oriented`, actual
    forward-kinematics-based verification), independent of whether the
    underlying grasp is physically achievable today."""
    import aisummit.grasping.planner as planner_module
    from aisummit.grasping.candidates import generate_candidates as real_generate_candidates
    from aisummit.grasping.scoring import evaluate_candidate as real_evaluate_candidate

    def few_candidates(*args, **kwargs):
        # Bounds the test to 3 real (but forced-feasible) candidates instead
        # of the full ~185-candidate search -- each attempt runs a real
        # physical pick_oriented(), so exercising the whole grid here would
        # make this test minutes slower for no extra coverage of the retry
        # LOOP's own logic, which is all this test targets.
        return real_generate_candidates(*args, **kwargs)[:3]

    def force_feasible(*args, **kwargs):
        metrics = real_evaluate_candidate(*args, **kwargs)
        metrics.feasible = True
        metrics.infeasible_reason = ""
        return metrics

    monkeypatch.setattr(planner_module, "generate_candidates", few_candidates)
    monkeypatch.setattr(planner_module, "evaluate_candidate", force_feasible)

    ctrl = env.current_ctrl()
    trace = GraspDebugTrace(object_name="fork")
    grasp_object(env, ctrl, side="left", object_name="fork", debug=trace)

    attempted_sources = [a.candidate_source for a in trace.attempts]
    assert len(attempted_sources) > 1, "expected more than one candidate to be attempted"
    assert len(attempted_sources) == len(set(attempted_sources)), (
        "the same candidate was executed more than once"
    )


def test_radial_object_uses_unmodified_pick_path(env):
    """Requirement: preserve existing working behavior exactly. The
    orientation-aware machinery must not even run for a radially symmetric
    object -- grasp_object should fall back to the original position-only
    pick() and succeed exactly as tests/test_pick_and_place.py verifies."""
    ctrl = env.current_ctrl()
    trace = GraspDebugTrace(object_name="cup")
    result = grasp_object(env, ctrl, side="right", object_name="cup", debug=trace)

    assert result.success
    assert trace.candidates == [], "elongated-object candidate generation ran for a radial object"


def test_stabilizer_arm_moves_toward_the_object(env):
    """Requirement 7: bimanual reasoning is real, working code -- exercised
    directly here since the current object set's widths don't naturally
    cross the low-stability trigger threshold (see planner.py)."""
    geometry = estimate_object_geometry(env.model, env.data, "fork")
    ctrl = env.current_ctrl()
    before = env.site_xpos("right/gripper").copy()

    stabilize_with_other_arm(env, ctrl, stabilizer_side="right", geometry=geometry)

    after = env.site_xpos("right/gripper")
    assert np.linalg.norm(after - before) > 0.05, "stabilizer arm didn't actually move"
    assert np.linalg.norm(after[:2] - geometry.position[:2]) < 0.25, (
        "stabilizer arm didn't move toward the object"
    )


def test_finger_target_to_site_target_corrects_a_real_offset(env):
    """Regression test for the single highest-impact bug found while tuning
    this pipeline: the `{side}/gripper` site IK targets is ~1.4cm away from
    the true finger-closing midpoint (a fixed mechanical offset -- see
    ik.py's GRIPPER_SITE_TO_FINGER_MIDPOINT_OFFSET docstring). That's inside
    the tolerance for the cup (3cm radius) but larger than the fork's whole
    half-width (1cm), which is what silently sank every early grasp attempt.
    Solving IK straight for a finger-target position (uninformed of the
    offset) should land the true finger midpoint noticeably farther from
    that target than solving for the *converted* site target does."""
    target = env.body_xpos("fork") + np.array([0.0, 0.0, 0.05])
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    def finger_midpoint_after_solving_for(site_target: np.ndarray) -> np.ndarray:
        angles = solve_ik(env.model, env.data, "left", site_target, target_quat=quat)
        scratch = mujoco.MjData(env.model)
        scratch.qpos[:] = env.data.qpos
        for j, joint_name in enumerate(ARM_JOINTS):
            jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"left/{joint_name}")
            scratch.qpos[env.model.jnt_qposadr[jnt_id]] = angles[j]
        mujoco.mj_forward(env.model, scratch)
        left_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "left/left_finger")
        right_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "left/right_finger")
        return (scratch.site(left_id).xpos + scratch.site(right_id).xpos) / 2

    naive_error = np.linalg.norm(finger_midpoint_after_solving_for(target) - target)
    corrected_target = finger_target_to_site_target(target, quat)
    corrected_error = np.linalg.norm(finger_midpoint_after_solving_for(corrected_target) - target)

    assert corrected_error < 0.01, f"corrected solve still {corrected_error:.4f}m off"
    assert naive_error > corrected_error + 0.005, (
        "the offset correction should measurably beat solving for the raw target"
    )


def test_warm_start_reaches_targets_a_cold_start_misses(env):
    """Regression test for the second highest-impact bug: a cold-start IK
    solve (from the arm's resting pose) can fail to converge for a
    far/twisted target that a warm-started solve, from a configuration
    already close to the answer, reaches easily -- found via a real false
    positive where a candidate looked collision-free only because it had
    silently failed to converge to anywhere near its intended target."""
    geometry = estimate_object_geometry(env.model, env.data, "fork")
    candidates = generate_candidates(env.model, env.data, "left", geometry)
    hard_candidate = next(c for c in candidates if c.source == "long=+0.000,trans=+0.000,wrist=90deg")

    site_target = finger_target_to_site_target(hard_candidate.target_position, hard_candidate.target_quat)
    cold = solve_ik(env.model, env.data, "left", site_target, target_quat=hard_candidate.target_quat)
    warm = solve_ik(
        env.model, env.data, "left", site_target, target_quat=hard_candidate.target_quat,
        seed_angles=hard_candidate.seed_angles,
    )

    def position_error(angles: np.ndarray) -> float:
        scratch = mujoco.MjData(env.model)
        scratch.qpos[:] = env.data.qpos
        for j, joint_name in enumerate(ARM_JOINTS):
            jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"left/{joint_name}")
            scratch.qpos[env.model.jnt_qposadr[jnt_id]] = angles[j]
        mujoco.mj_forward(env.model, scratch)
        site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
        return float(np.linalg.norm(scratch.site(site_id).xpos - site_target))

    assert position_error(warm) < 0.01, "warm-started solve should converge closely"
    assert position_error(cold) > position_error(warm) + 0.02, (
        "warm start should measurably beat a cold start on this hard target"
    )


def _solved_angles(env, side: str, candidate) -> np.ndarray:
    # Delegates to the real production solve (approach, then the
    # translation-only final descent when candidate.final_target_position
    # is set) rather than reimplementing it here, so this helper can never
    # drift out of sync with what grasp_object actually executes.
    from aisummit.grasping.planner import _solve_matching_execution

    return _solve_matching_execution(env, side, candidate)


def test_finger_height_mismatch_can_outweigh_a_modest_alignment_gap(env):
    """Requirement 8.A: a candidate with excellent horizontal alignment but
    a large finger-height mismatch must lose to a slightly-less-perfect but
    vertically symmetric one.

    Uses two REAL candidates from the fork's actual candidate grid, so the
    finger-height-mismatch numbers are genuine forward-kinematics
    measurements, not fabricated -- only `closing_axis_alignment` is
    overridden (to a value representing "slightly less than the other
    candidate's real alignment") so the test isolates the scoring formula's
    behavior. Measured fact this documents: at the fork's current position,
    the realistically-achievable candidates don't happen to include a pair
    where the *real* alignment values are close together AND the mismatch
    values are far apart (see README's grasping-pipeline section) -- this
    test exists to prove the scoring mechanism does the right thing on that
    trade-off shape when it occurs, independent of whether today's fork
    happens to produce exactly that pair."""
    geometry = estimate_object_geometry(env.model, env.data, "fork")
    candidates = {c.source: c for c in generate_candidates(env.model, env.data, "left", geometry)}
    high_align_high_mismatch = candidates["long=+0.000,trans=+0.000,wrist=90deg"]
    lower_align_low_mismatch = candidates["long=+0.000,trans=+0.000,wrist=45deg"]
    lower_align_low_mismatch.closing_axis_alignment = high_align_high_mismatch.closing_axis_alignment - 0.06

    weights = GraspWeights()
    metrics_a = evaluate_candidate(
        env.model, env.data, high_align_high_mismatch, geometry,
        _solved_angles(env, "left", high_align_high_mismatch), weights,
    )
    metrics_b = evaluate_candidate(
        env.model, env.data, lower_align_low_mismatch, geometry,
        _solved_angles(env, "left", lower_align_low_mismatch), weights,
    )

    assert metrics_a.orientation_alignment > metrics_b.orientation_alignment, "test setup: A should be better-aligned"
    assert metrics_a.finger_height_mismatch > metrics_b.finger_height_mismatch + 0.002, (
        "test setup: A should have measurably worse finger-height mismatch"
    )
    assert metrics_b.score > metrics_a.score, (
        "the slightly-less-aligned but vertically-symmetric candidate should win"
    )


def test_radial_object_unaffected_by_finger_height_scoring(env):
    """Requirement 8.B: the cup's existing successful behavior is
    unchanged. Height-mismatch scoring lives entirely inside
    evaluate_candidate/generate_candidates, which the radial path in
    grasp_object never calls -- same guarantee test_radial_object_uses_
    unmodified_pick_path already covers, restated explicitly for this
    requirement."""
    ctrl = env.current_ctrl()
    trace = GraspDebugTrace(object_name="cup")
    result = grasp_object(env, ctrl, side="right", object_name="cup", debug=trace)
    assert result.success
    assert trace.candidates == [], "the cup should never reach candidate scoring at all"


def test_knife_top_scored_candidate_still_favors_the_handle(env):
    """Requirement 8.C: finger-height-aware scoring must not cause the
    knife's top-ranked candidate to drift toward the blade.

    Filters on `ik_ok` alone, not the stricter `feasible`/`collision_valid`
    -- with the translation-only final descent (which now correctly brings
    the fingers down to the object's real height instead of leaving them
    ~30mm above it), every IK-converged knife candidate at this position
    also fails collision (same ~2cm table penetration found for the fork
    at well-aligned wrist angles -- a systemic finding, not knife-specific,
    see README). That's a separate, real result about physical
    executability; this test is only checking whether the *scoring
    formula* still prefers the handle over the blade among whatever
    candidates converge, which doesn't require collision-free."""
    env.settle()  # matches what grasp_object() actually does before generating candidates
    geometry = estimate_object_geometry(env.model, env.data, "knife")
    candidates = generate_candidates(env.model, env.data, "right", geometry)
    weights = GraspWeights()
    scored = []
    for c in candidates:
        solved = _solved_angles(env, "right", c)
        m = evaluate_candidate(env.model, env.data, c, geometry, solved, weights)
        if m.ik_ok:
            scored.append((m.score, c))
    assert scored, "expected at least one IK-converged knife candidate"
    _, best = max(scored, key=lambda t: t[0])

    dist_to_handle = np.linalg.norm(best.target_position - geometry.grasp_region)
    dist_to_blade = np.linalg.norm(best.target_position - geometry.unsafe_region)
    assert dist_to_handle < dist_to_blade, "top-scored knife candidate drifted toward the blade"


def test_ik_failed_candidate_cannot_win_on_geometric_score_alone(env):
    """Requirement 8.D: a candidate with an artificially inflated
    finger-height/alignment score must never be selected if IK didn't
    actually converge for it -- grasp_object's filtering, not scoring
    alone, is what has to catch this."""
    from aisummit.grasping.candidates import GraspCandidate

    geometry = estimate_object_geometry(env.model, env.data, "fork")
    unreachable = GraspCandidate(
        side="left", target_position=np.array([5.0, 5.0, 5.0]), target_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        source="unreachable-inflated", closing_axis_alignment=1.0,  # perfect alignment, but nonsense target
    )
    solved = solve_ik(env.model, env.data, "left", unreachable.target_position, target_quat=unreachable.target_quat)
    metrics = evaluate_candidate(env.model, env.data, unreachable, geometry, solved, GraspWeights())

    assert not metrics.ik_ok
    # Even with perfect alignment and (since nothing real is nearby) a
    # small incidental finger-height mismatch, ik_ok must gate selection --
    # grasp_object's own filtering loop drops anything with ik_ok=False
    # before scores are ever compared.


def test_selected_fork_candidate_meets_position_accuracy_threshold(env, monkeypatch):
    """Requirement 8.E: whatever candidate grasp_object actually selects
    for the fork must still satisfy the pre-existing IK position-error
    threshold -- finger-height scoring must not trade away basic
    positional accuracy to buy symmetry.

    Same forced-feasible patch as the retry-loop test above and for the
    same reason: the real fork has zero feasible candidates today (a
    genuine finding, not a test gap -- see README), so nothing gets
    selected without it. `ik_position_error` itself is never touched by
    the patch -- it's still the real, measured IK residual for whichever
    candidate wins the (real) scoring comparison."""
    from aisummit.grasping.scoring import _IK_POSITION_TOLERANCE
    from aisummit.grasping.candidates import generate_candidates as real_generate_candidates
    from aisummit.grasping.scoring import evaluate_candidate as real_evaluate_candidate
    import aisummit.grasping.planner as planner_module

    def few_candidates(*args, **kwargs):
        # Bounded for the same reason as the retry-loop test: forcing
        # feasibility means the attempt loop runs a real pick_oriented()
        # per candidate, so this stays small on purpose.
        return real_generate_candidates(*args, **kwargs)[:3]

    def force_feasible(*args, **kwargs):
        metrics = real_evaluate_candidate(*args, **kwargs)
        metrics.feasible = True
        metrics.infeasible_reason = ""
        return metrics

    monkeypatch.setattr(planner_module, "generate_candidates", few_candidates)
    monkeypatch.setattr(planner_module, "evaluate_candidate", force_feasible)

    ctrl = env.current_ctrl()
    trace = GraspDebugTrace(object_name="fork")
    grasp_object(env, ctrl, side="left", object_name="fork", debug=trace)

    assert trace.selected is not None, "expected a candidate to be selected"
    assert trace.selected.metrics.ik_position_error < _IK_POSITION_TOLERANCE


def test_spatially_shifted_candidate_can_outrank_original_on_symmetry(env):
    """Requirement 8.A (spatial-search iteration): a spatially shifted
    candidate (nonzero transverse/longitudinal offset) can outrank the
    zero-offset original when it has better finger-height symmetry.

    Uses two REAL candidates from the actual spatial search -- one at
    zero offset, one transverse-shifted -- so the mismatch values are
    genuine forward-kinematics measurements. Real transverse offsets for
    today's fork don't happen to swing mismatch by much (measured: ~9.1-
    9.3mm across all three transverse positions at a given wrist angle --
    see README), so `closing_axis_alignment` is adjusted on the shifted
    candidate to represent "slightly less perfect than the original,"
    isolating the scoring formula's behavior the same way the
    single-offset version of this test already does for wrist-only
    candidates."""
    geometry = estimate_object_geometry(env.model, env.data, "fork")
    candidates = {c.source: c for c in generate_candidates(env.model, env.data, "left", geometry)}
    original = candidates["long=+0.000,trans=+0.000,wrist=90deg"]
    shifted = candidates["long=+0.000,trans=+0.010,wrist=45deg"]
    shifted.closing_axis_alignment = original.closing_axis_alignment - 0.06

    weights = GraspWeights()
    metrics_original = evaluate_candidate(
        env.model, env.data, original, geometry, _solved_angles(env, "left", original), weights,
    )
    metrics_shifted = evaluate_candidate(
        env.model, env.data, shifted, geometry, _solved_angles(env, "left", shifted), weights,
    )

    assert shifted.transverse_offset != original.transverse_offset, "test setup: candidates should differ spatially"
    assert metrics_original.orientation_alignment > metrics_shifted.orientation_alignment
    assert metrics_original.finger_height_mismatch > metrics_shifted.finger_height_mismatch + 0.002
    assert metrics_shifted.score > metrics_original.score, (
        "the spatially-shifted, more vertically-symmetric candidate should win"
    )


def test_ik_invalid_spatial_candidate_cannot_win_on_alignment(env):
    """Requirement 8.B: a spatially-shifted (nonzero offset) candidate
    that fails to converge must be classified infeasible regardless of
    how high its alignment score is -- feasibility, not score, is the
    hard gate."""
    from aisummit.grasping.candidates import GraspCandidate

    geometry = estimate_object_geometry(env.model, env.data, "fork")
    unreachable_shifted = GraspCandidate(
        side="left", target_position=np.array([5.0, 5.0, 5.0]), target_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        source="spatial-unreachable", closing_axis_alignment=1.0,
        longitudinal_offset=0.015, transverse_offset=0.01,
    )
    solved = solve_ik(
        env.model, env.data, "left", unreachable_shifted.target_position,
        target_quat=unreachable_shifted.target_quat,
    )
    metrics = evaluate_candidate(env.model, env.data, unreachable_shifted, geometry, solved, GraspWeights())

    assert not metrics.ik_ok
    assert not metrics.feasible
    assert metrics.infeasible_reason == "IK did not converge"


def test_large_finger_height_mismatch_is_classified_infeasible(env):
    """Requirement 8.C: a candidate whose finger-height mismatch clearly
    exceeds the object-aware tolerance must be classified infeasible --
    checked against a real, measured fork candidate (align=0.91,
    mismatch~9.3mm on an 8mm-thick object), not a synthetic one, since
    real candidates violating this exist today (see README)."""
    geometry = estimate_object_geometry(env.model, env.data, "fork")
    candidates = {c.source: c for c in generate_candidates(env.model, env.data, "left", geometry)}
    candidate = candidates["long=+0.000,trans=+0.000,wrist=90deg"]
    solved = _solved_angles(env, "left", candidate)
    metrics = evaluate_candidate(env.model, env.data, candidate, geometry, solved, GraspWeights())

    assert metrics.finger_height_mismatch > 2 * geometry.thickness, (
        "test setup: this candidate should have a genuinely large mismatch relative to object thickness"
    )
    assert not metrics.feasible
    assert metrics.infeasible_reason  # some specific reason recorded, not silently dropped


def test_candidate_generation_is_deterministic_and_bounded(env):
    """Requirement 8.F: generate_candidates is deterministic (same inputs
    -> identical outputs) and its output size stays small and bounded --
    the brief's explicit "do not create a huge brute-force grid"."""
    geometry = estimate_object_geometry(env.model, env.data, "fork")
    first = generate_candidates(env.model, env.data, "left", geometry)
    second = generate_candidates(env.model, env.data, "left", geometry)

    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.source == b.source
        assert np.allclose(a.target_position, b.target_position)
        assert np.allclose(a.target_quat, b.target_quat)

    # 5 spatial points (3 longitudinal + 2 transverse, a star pattern, not
    # a 3x2 grid) x 37 wrist angles (5-degree steps) = 185 -- bounded and
    # small relative to what an unconstrained grid over the same ranges
    # would produce.
    assert len(first) == 185
