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

Spatial search (added after the wrist-only search hit a wall -- see
README's "spatial repositioning" section): the wrist sweep alone changes
*orientation* around a fixed grasp point, but measurement showed the
~9mm finger-height mismatch near the well-aligned region was constant
across wrist angles, grasp heights, and both arms at the *current* grasp
point -- a real kinematic coupling at that specific reach, not something
orientation search alone can escape. `generate_candidates` now also
varies the grasp POINT itself by a small, bounded set of horizontal
offsets (longitudinal + transverse, using the object-relative axes
`geometry.principal_axis` and `geometry.transverse_axis_h` already
exposed by geometry.py) before running the same wrist sweep at each one.
This is a star pattern (each offset axis varied independently around the
base point), not a full 2D grid, to keep the search bounded: 5 spatial
points x 37 wrist angles, not 3x3x37. World-frame X/Y offsets are not
swept as a separate dimension: for a box object with zero rotation (the
fork/knife's authored pose), the principal/transverse axes already ARE
world Y/X, so a separate world-frame sweep would just re-run the same
numbers under different labels -- the object-relative axes are the
general mechanism (they still work if that assumption stops holding for
a future rotated object), not a fork-specific shortcut.

Approach point vs. real close point: `target_position` includes
`_GRASP_HEIGHT_CLEARANCE`, needed for the gripper BODY to clear the table
at wide wrist angles -- but that same clearance was found to also lift
the FINGERS well above the object, since `pick_oriented` never used to
come back down before closing (measured directly: ~30mm above the fork's
real surface, for every candidate, regardless of wrist angle or spatial
offset). Each candidate now also carries `final_target_position` -- the
same horizontal point at the object's real, un-elevated height --
which `pick_oriented`'s translation-only final descent uses to actually
close on the object instead of well above it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aisummit.control.ik import solve_ik
from aisummit.grasping.geometry import ObjectGeometry, transverse_axis_h
from aisummit.sim.env import ARM_JOINTS

_LONGITUDINAL_OFFSETS = (-0.015, 0.0, 0.015)  # meters, along the object's principal axis
# Small and separate from the longitudinal set on purpose (a star pattern,
# not a grid) -- see module docstring's "Spatial search" section for why
# this dimension exists at all: wrist orientation alone plateaus at a
# fixed ~9mm finger-height mismatch regardless of angle, so the search now
# also asks whether a nearby grasp POINT changes that coupling.
_TRANSVERSE_OFFSETS = (-0.01, 0.01)  # meters, perpendicular to the object's principal axis
# The authored handle site sits at the object's exact centerline height,
# right at the tabletop -- fine for *where* to grasp, but it leaves the
# gripper's body almost no vertical clearance before wider wrist angles
# drive it into the table (found empirically: several otherwise-good
# candidates were rejected for 1-5cm table penetration). Lifting the
# target a little is a grasp-planning decision, not a geometry-estimation
# one, so it lives here rather than adjusting the authored handle height.
_GRASP_HEIGHT_CLEARANCE = 0.03
# 5-degree steps rather than 15: at 15-degree resolution, the only nearby
# sample points to the good-alignment region (75, 90) either penetrate the
# table or -- found the hard way -- fail to converge to anywhere near the
# target at all (a candidate that LOOKED like a great compromise, good
# alignment with near-zero finger-height mismatch, turned out on inspection
# to be a completely different, non-converged arm pose off in space; its
# "good" mismatch was meaningless because the whole solve was wrong). At
# 5-degree resolution a genuinely convergent, collision-free point with
# both decent alignment and low finger-height mismatch exists nearby.
# Still cheap and deterministic FK/IK, not an LLM call -- ~3x more
# candidates, not an exploded search space.
_WRIST_SWEEP_DEG = tuple(range(0, 181, 5))  # degrees; 180 deg-periodic for a parallel gripper


def _spatial_offsets(
    longitudinal_offsets: tuple[float, ...], transverse_offsets: tuple[float, ...]
) -> list[tuple[float, float]]:
    """Star pattern around (0, 0): every longitudinal offset at zero
    transverse, plus every nonzero transverse offset at zero longitudinal.
    5 points for the current defaults (3 + 2), not the 3x2=6-plus-overlap
    a full grid would give -- deliberately small per the brief's "do not
    create a huge brute-force grid"."""
    points = [(lon, 0.0) for lon in longitudinal_offsets]
    points += [(0.0, trans) for trans in transverse_offsets if trans != 0.0]
    return points


@dataclass
class GraspCandidate:
    side: str
    target_position: np.ndarray  # approach point -- includes _GRASP_HEIGHT_CLEARANCE for table clearance
    target_quat: np.ndarray
    source: str  # human-readable provenance, e.g. "long=+0.015,trans=+0.000,wrist=90deg"
    closing_axis_alignment: float  # 0..1, 1 = perfectly perpendicular to the object's axis
    longitudinal_offset: float = 0.0  # meters, along the object's principal axis
    transverse_offset: float = 0.0  # meters, along the object's transverse axis
    final_target_position: np.ndarray | None = None  # the real close point, no clearance -- see control/primitives.py::pick_oriented
    seed_angles: np.ndarray | None = None  # warm-start for the final 6-DOF solve -- see solve_ik's docstring


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
    longitudinal_offsets: tuple[float, ...] = _LONGITUDINAL_OFFSETS,
    transverse_offsets: tuple[float, ...] = _TRANSVERSE_OFFSETS,
    wrist_sweep_deg: tuple[float, ...] = _WRIST_SWEEP_DEG,
) -> list[GraspCandidate]:
    real_close_point = geometry.grasp_region if geometry.grasp_region is not None else geometry.position
    base_point = real_close_point + np.array([0, 0, _GRASP_HEIGHT_CLEARANCE])
    axis_h = _horizontal_unit(geometry.principal_axis)
    trans_h = transverse_axis_h(geometry)
    wrist_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/wrist_rotate")
    wrist_qadr = model.jnt_qposadr[wrist_jnt_id]

    candidates: list[GraspCandidate] = []
    for lon_offset, trans_offset in _spatial_offsets(longitudinal_offsets, transverse_offsets):
        horizontal_offset = lon_offset * axis_h + trans_offset * trans_h
        target_pos = base_point + horizontal_offset
        # Same horizontal position as the approach point, at the object's
        # real height instead of the clearance-elevated one -- what
        # pick_oriented's translation-only final descent actually closes
        # on (see its docstring for why the approach point alone isn't the
        # real close height).
        final_target_pos = real_close_point + horizontal_offset
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

            wrist_overridden_angles = np.array(
                [scratch.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{j}")]]
                 for j in ARM_JOINTS]
            )

            candidates.append(
                GraspCandidate(
                    side=side,
                    target_position=target_pos,
                    target_quat=quat,
                    source=f"long={lon_offset:+.3f},trans={trans_offset:+.3f},wrist={wrist_deg}deg",
                    closing_axis_alignment=alignment,
                    longitudinal_offset=lon_offset,
                    transverse_offset=trans_offset,
                    final_target_position=final_target_pos,
                    seed_angles=wrist_overridden_angles,
                )
            )
    return candidates
