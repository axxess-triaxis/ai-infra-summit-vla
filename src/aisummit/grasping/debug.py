"""Observability for the grasp pipeline: a structured trace of every
candidate considered (accepted or rejected, and why), plus an optional
visual overlay projecting the principal axis and chosen grasp frame onto
the overhead camera image for the demo recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from PIL import ImageDraw

from aisummit.grasping.geometry import ObjectGeometry
from aisummit.grasping.scoring import GraspMetrics


@dataclass
class CandidateTrace:
    source: str
    metrics: GraspMetrics | None
    accepted: bool
    rejection_reason: str = ""


@dataclass
class AttemptTrace:
    candidate_source: str
    verified: bool
    note: str = ""


@dataclass
class GraspDebugTrace:
    object_name: str
    geometry: ObjectGeometry | None = None
    candidates: list[CandidateTrace] = field(default_factory=list)
    selected: CandidateTrace | None = None
    attempts: list[AttemptTrace] = field(default_factory=list)
    verbose: bool = False

    def log(self, *parts: object) -> None:
        if self.verbose:
            print("[grasp]", self.object_name, *parts)

    def record_candidate(self, trace: CandidateTrace) -> None:
        self.candidates.append(trace)
        reason = trace.rejection_reason or "ok"
        score = f"{trace.metrics.score:.3f}" if trace.metrics else "n/a"
        self.log(f"candidate {trace.source}: score={score} accepted={trace.accepted} ({reason})")

    def record_attempt(self, attempt: AttemptTrace) -> None:
        self.attempts.append(attempt)
        self.log(f"attempt with {attempt.candidate_source}: verified={attempt.verified} {attempt.note}")

    def summary(self) -> str:
        lines = [f"Grasp trace for {self.object_name!r}:"]
        for c in self.candidates:
            tag = "SELECTED" if self.selected is c else ("accepted" if c.accepted else "rejected")
            lines.append(f"  [{tag}] {c.source}: {c.rejection_reason or (c.metrics and f'score={c.metrics.score:.3f}')}")
        for a in self.attempts:
            lines.append(f"  attempt({a.candidate_source}) -> verified={a.verified} {a.note}")
        return "\n".join(lines)


def world_to_pixel(
    model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, point: np.ndarray, width: int, height: int
) -> tuple[int, int] | None:
    """Projects a world-frame point into pixel coordinates for the named
    camera, using MuJoCo's own camera convention (looks down local -Z).
    Returns None if the point is behind the camera."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)  # columns = camera axes in world frame
    p_cam = cam_rot.T @ (np.asarray(point) - cam_pos)
    depth = -p_cam[2]
    if depth <= 1e-6:
        return None
    fovy_rad = np.radians(model.cam_fovy[cam_id])
    focal = 0.5 * height / np.tan(0.5 * fovy_rad)
    px = width / 2 + focal * (p_cam[0] / depth)
    py = height / 2 - focal * (p_cam[1] / depth)
    return int(px), int(py)


def draw_grasp_overlay(
    image: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    geometry: ObjectGeometry,
    selected_position: np.ndarray | None = None,
) -> np.ndarray:
    from PIL import Image

    height, width = image.shape[0], image.shape[1]
    pil_img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_img)

    axis_start = geometry.position - 0.5 * geometry.half_length * geometry.principal_axis
    axis_end = geometry.position + 0.5 * geometry.half_length * geometry.principal_axis
    p0 = world_to_pixel(model, data, camera_name, axis_start, width, height)
    p1 = world_to_pixel(model, data, camera_name, axis_end, width, height)
    if p0 and p1:
        draw.line([p0, p1], fill=(255, 255, 0), width=3)

    if geometry.grasp_region is not None:
        p = world_to_pixel(model, data, camera_name, geometry.grasp_region, width, height)
        if p:
            draw.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], outline=(0, 255, 0), width=2)

    if geometry.unsafe_region is not None:
        p = world_to_pixel(model, data, camera_name, geometry.unsafe_region, width, height)
        if p:
            draw.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], outline=(255, 0, 0), width=2)

    if selected_position is not None:
        p = world_to_pixel(model, data, camera_name, selected_position, width, height)
        if p:
            draw.rectangle([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], outline=(0, 200, 255), width=2)

    return np.array(pil_img)
