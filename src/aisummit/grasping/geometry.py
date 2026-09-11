"""Object geometry + grasp-region estimation.

Perception scoping note: this reads MuJoCo's own ground-truth body pose and
geom size/type rather than running segmentation + PCA on the rendered image.
That's a deliberate hackathon-week trade-off, not a shortcut pretending to be
something it isn't -- it stands in for the
"segmentation-mask PCA / oriented bounding box / depth point-cloud PCA"
stage the brief describes, using privileged simulator state instead of
pixels. The interface (`estimate_object_geometry` returning position +
principal_axis + grasp_region) is exactly what a real vision module would
also need to produce, so swapping in real segmentation+PCA later only means
replacing this one function's internals, not anything downstream of it.

Handle/blade regions similarly come from authored MJCF sites
(`{object}/handle`, `{object}/unsafe`) rather than a learned grasp-region
detector -- again, a stand-in with a clean swap-in point, not a fabricated
capability.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Ratio of (longest half-extent / next-longest half-extent) above which a box
# is treated as elongated (needs orientation-aware grasping) rather than
# "compact enough to grasp from any angle".
_ELONGATION_RATIO = 2.0


@dataclass
class ObjectGeometry:
    name: str
    position: np.ndarray  # world xyz of the body frame (m)
    orientation: np.ndarray  # world quat, wxyz (MuJoCo convention)
    principal_axis: np.ndarray  # unit vector, world frame
    half_length: float  # half-extent along principal_axis (m)
    radius: float  # characteristic cross-section half-extent/radius (m)
    thickness: float  # half-extent along the SHORTEST axis (m) -- see below
    is_elongated: bool
    grasp_region: np.ndarray | None  # world xyz of the authored handle site, if any
    grasp_region_confidence: float  # 1.0 if an authored handle site exists, else 0.5 (CoM fallback)
    unsafe_region: np.ndarray | None  # world xyz of a blade/unsafe site, if any


def transverse_axis_h(geometry: ObjectGeometry) -> np.ndarray:
    """Unit horizontal vector perpendicular to the object's principal axis
    -- its "width" direction. Used by grasp candidate search (candidates.py)
    to test spatial offsets that aren't purely along the object's length,
    reusing the same axis abstraction rather than inventing a second one."""
    axis = geometry.principal_axis.copy()
    axis[2] = 0.0
    norm = np.linalg.norm(axis)
    axis_h = axis / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    return np.array([-axis_h[1], axis_h[0], 0.0])


def _site_world_pos(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray | None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        return None
    return data.site(site_id).xpos.copy()


def estimate_object_geometry(
    model: mujoco.MjModel, data: mujoco.MjData, object_name: str
) -> ObjectGeometry:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id == -1:
        raise ValueError(f"No body named {object_name!r} in model")
    position = data.xpos[body_id].copy()
    orientation = data.xquat[body_id].copy()

    geom_ids = np.nonzero(model.geom_bodyid == body_id)[0]
    if len(geom_ids) == 0:
        raise ValueError(f"Body {object_name!r} has no geoms")
    geom_id = geom_ids[0]
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        local_axis = np.array([0.0, 0.0, 1.0])  # cylinder axis is local z
        half_length = float(size[1])
        radius = float(size[0])
        # A cylinder has no distinct "flat" dimension the way a box does --
        # its cross-section is round in every direction perpendicular to
        # the axis. No elongated cylinder exists in the current object set
        # to calibrate against, so this falls back to radius (a rod's
        # cross-section is at least `radius` wide in any direction, so a
        # same-order finger-height tolerance is a reasonable default, not
        # a verified one).
        thickness = radius
        is_elongated = half_length > _ELONGATION_RATIO * radius
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        order = np.argsort(size)[::-1]  # longest half-extent first
        longest, second = size[order[0]], size[order[1]]
        local_axis = np.zeros(3)
        local_axis[order[0]] = 1.0
        half_length = float(longest)
        radius = float(second)
        thickness = float(size[order[2]])  # shortest half-extent -- the fork/knife's flat dimension
        is_elongated = longest > _ELONGATION_RATIO * second
    else:
        raise ValueError(f"Unsupported geom type {geom_type} for {object_name!r}")

    world_axis = np.zeros(3)
    mujoco.mju_rotVecQuat(world_axis, local_axis, orientation)

    grasp_region = _site_world_pos(model, data, f"{object_name}/handle")
    unsafe_region = _site_world_pos(model, data, f"{object_name}/unsafe")

    return ObjectGeometry(
        name=object_name,
        position=position,
        orientation=orientation,
        principal_axis=world_axis,
        half_length=half_length,
        radius=radius,
        thickness=thickness,
        is_elongated=is_elongated,
        grasp_region=grasp_region,
        grasp_region_confidence=1.0 if grasp_region is not None else 0.5,
        unsafe_region=unsafe_region,
    )
