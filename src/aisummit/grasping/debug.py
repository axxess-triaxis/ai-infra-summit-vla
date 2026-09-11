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
        if trace.metrics is None:
            self.log(f"candidate {trace.source}: score=n/a accepted={trace.accepted} ({reason})")
            return
        m = trace.metrics
        self.log(
            f"candidate={trace.source} object={self.object_name} "
            f"position_error_mm={m.ik_position_error * 1000:.1f} "
            f"horizontal_alignment={m.orientation_alignment:.2f} "
            f"finger_left_z={m.finger_left_z:.4f} finger_right_z={m.finger_right_z:.4f} "
            f"finger_height_mismatch_mm={m.finger_height_mismatch * 1000:.1f} "
            f"clearance_ok={m.collision_valid} IK_success={m.ik_ok} "
            f"total_score={m.score:.3f} accepted={trace.accepted} reason={reason}"
        )

    def record_attempt(self, attempt: AttemptTrace) -> None:
        self.attempts.append(attempt)
        self.log(f"attempt with {attempt.candidate_source}: verified={attempt.verified} {attempt.note}")

    def print_selected_summary(self) -> None:
        """Prints the fixed-format block requested for the selected grasp,
        e.g. for the fork:
            FORK GRASP
            horizontal alignment: 0.91
            finger height mismatch: 9.3 mm
            position error: 0.5 mm
            IK: SUCCESS
            collision: CLEAR
            candidate score: 0.734
        No-op if nothing was selected."""
        if self.selected is None or self.selected.metrics is None:
            return
        m = self.selected.metrics
        print(f"{self.object_name.upper()} GRASP")
        print(f"  horizontal alignment: {m.orientation_alignment:.2f}")
        print(f"  finger height mismatch: {m.finger_height_mismatch * 1000:.1f} mm")
        print(f"  position error: {m.ik_position_error * 1000:.1f} mm")
        print(f"  IK: {'SUCCESS' if m.ik_ok else 'FAILED'}")
        print(f"  collision: {'CLEAR' if m.collision_valid else 'BLOCKED'}")
        print(f"  candidate score: {m.score:.3f}")

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
