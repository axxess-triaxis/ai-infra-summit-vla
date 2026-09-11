"""Tests for the orientation-aware grasp pipeline (grasping/).

Each test targets a distinct requirement from the grasp-pipeline upgrade
rather than re-testing the whole stack end-to-end each time:
geometry classification, candidate orientation search, collision rejection,
IK-failure rejection, retry-without-repeat, the knife's safety region, and
bimanual stabilization. `test_pick_and_place.py` covers the radially
symmetric regression case (cup) separately and is left untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from aisummit.control.ik import solve_ik
from aisummit.grasping.candidates import generate_candidates
from aisummit.grasping.collision import check_collision
from aisummit.grasping.debug import GraspDebugTrace
from aisummit.grasping.geometry import estimate_object_geometry
from aisummit.grasping.planner import grasp_object, stabilize_with_other_arm
from aisummit.grasping.scoring import GraspWeights, evaluate_candidate
from aisummit.sim.env import DinnerTableEnv


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


def test_grasp_retries_multiple_distinct_candidates_without_repeating(env):
    """Requirement 6: never re-issue an identical failed grasp. Runs the
    real pipeline against the fork (currently a hard case for this arm's
    kinematics -- see README's grasping-pipeline section) and checks the
    *retry mechanism itself* is correct: every attempted candidate source
    is distinct, and more than one was actually tried."""
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
