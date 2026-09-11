"""Static collision/validity check for a candidate arm pose.

Poses the arm at a candidate's solved joint config on a scratch MjData (no
dynamics stepped) and asks MuJoCo's own contact detection whether that pose
touches anything it shouldn't -- another tableware object, or the table at
an unexpected point. Contact with the intended target object, and with the
table underneath a normal approach, is expected and excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from aisummit.sim.env import ARM_JOINTS, TABLEWARE


def _side_body_ids(model: mujoco.MjModel, side: str) -> set[int]:
    ids = set()
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith(f"{side}/"):
            ids.add(i)
    return ids


_WORLD_BODY_ID = 0
# 1mm of table/floor penetration is treated as a real collision, not
# floating-point/contact-softness noise from resting near a flat object.
_TABLE_PENETRATION_TOLERANCE = -0.001


def check_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    arm_angles: np.ndarray,
    target_object: str,
    clearance_margin: float = 0.0,
) -> tuple[bool, str]:
    """Returns (is_valid, reason). reason is empty when valid."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    for j, joint_name in enumerate(ARM_JOINTS):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{joint_name}")
        scratch.qpos[model.jnt_qposadr[jnt_id]] = arm_angles[j]
    mujoco.mj_forward(model, scratch)

    side_bodies = _side_body_ids(model, side)
    other_objects = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                      for name in TABLEWARE if name != target_object}

    for i in range(scratch.ncon):
        contact = scratch.contact[i]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        bodies = {body1, body2}
        if not (bodies & side_bodies):
            continue  # doesn't involve this arm at all

        hit_other_object = bodies & other_objects
        if hit_other_object and contact.dist <= clearance_margin:
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, next(iter(hit_other_object)))
            return False, f"arm would collide with {other_name!r} (dist={contact.dist:.4f})"

        if _WORLD_BODY_ID in bodies and contact.dist < _TABLE_PENETRATION_TOLERANCE:
            return False, f"arm would penetrate the table/floor (dist={contact.dist:.4f})"

    return True, ""


@dataclass
class Contact:
    body_a: str
    body_b: str
    dist: float
    kind: str  # "intended_target" | "intended_support" | "table_penetration" | "unintended_collision"


def classify_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    arm_angles: np.ndarray,
    target_object: str,
    intended_support_bodies: frozenset[str] = frozenset(),
) -> list[Contact]:
    """Full-detail counterpart to `check_collision`: classifies every
    contact the arm is party to instead of short-circuiting on the first
    disqualifying one. "Contact is not collision" -- a strategy that
    intentionally rests against the target object, or a designated
    support/backstop body (e.g. the plate, for a future scooping
    strategy), needs to tell that apart from an unintended collision or
    real table penetration. `check_collision` itself is untouched -- every
    existing caller keeps its exact current behavior; this is purely
    additive, used by `grasping/contact_strategies.py`."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    for j, joint_name in enumerate(ARM_JOINTS):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{joint_name}")
        scratch.qpos[model.jnt_qposadr[jnt_id]] = arm_angles[j]
    mujoco.mj_forward(model, scratch)

    side_bodies = _side_body_ids(model, side)
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_object)
    support_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in intended_support_bodies
    }

    contacts: list[Contact] = []
    for i in range(scratch.ncon):
        contact = scratch.contact[i]
        body1, body2 = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        bodies = {body1, body2}
        if not (bodies & side_bodies):
            continue

        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "?"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "?"

        if target_body_id in bodies:
            kind = "intended_target"
        elif bodies & support_body_ids:
            kind = "intended_support"
        elif _WORLD_BODY_ID in bodies and contact.dist < _TABLE_PENETRATION_TOLERANCE:
            kind = "table_penetration"
        elif contact.dist <= 0.0:
            kind = "unintended_collision"
        else:
            continue  # not actually touching, just close -- not worth classifying
        contacts.append(Contact(body_a=name1, body_b=name2, dist=float(contact.dist), kind=kind))
    return contacts
