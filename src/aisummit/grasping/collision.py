"""Static collision/validity check for a candidate arm pose.

Poses the arm at a candidate's solved joint config on a scratch MjData (no
dynamics stepped) and asks MuJoCo's own contact detection whether that pose
touches anything it shouldn't -- another tableware object, or the table at
an unexpected point. Contact with the intended target object, and with the
table underneath a normal approach, is expected and excluded.
"""

from __future__ import annotations

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
