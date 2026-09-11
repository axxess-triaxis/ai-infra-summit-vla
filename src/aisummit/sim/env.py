"""ALOHA bimanual dinner-table environment.

Wraps the raw `mujoco` Python bindings directly (no robosuite -- see README
for why: robosuite's OSC controller is incompatible with every current
mujoco release on Windows). The ALOHA arms use position actuators, so
control is just writing target joint angles to `data.ctrl`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
SCENE_PATH = _REPO_ROOT / "assets" / "mujoco_menagerie" / "aloha" / "dinner_table.xml"

ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
SIDES = ("left", "right")
TABLEWARE = ("plate", "fork", "knife", "cup")


@dataclass
class Observation:
    qpos: np.ndarray
    qvel: np.ndarray
    object_positions: dict[str, np.ndarray]
    gripper_positions: dict[str, np.ndarray]
    image: np.ndarray | None = None


class DinnerTableEnv:
    def __init__(self, render_width: int = 640, render_height: int = 480):
        if not SCENE_PATH.exists():
            raise FileNotFoundError(
                f"Scene not found at {SCENE_PATH}. Did the mujoco_menagerie "
                "submodule/clone under assets/ get removed?"
            )
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self._neutral_key = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose"
        )
        self._rng = np.random.default_rng()

    def reset(self, randomize: bool = True, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._neutral_key)
        # mj_resetDataKeyframe pads a keyframe recorded before these free
        # joints existed with zeros, not each body's authored default pose --
        # restore the real initial pose from qpos0 for every tableware object.
        for name in TABLEWARE:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            qadr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qadr : qadr + 7] = self.model.qpos0[qadr : qadr + 7]
            if randomize:
                dx, dy = self._rng.uniform(-0.03, 0.03, size=2)
                self.data.qpos[qadr] += dx
                self.data.qpos[qadr + 1] += dy
        mujoco.mj_forward(self.model, self.data)
        return self._observe()

    def settle(self, steps: int = 60) -> Observation:
        """Steps physics with the arms held at their current ctrl so
        free-jointed objects reach their true resting contact height.
        Authored object heights aren't exactly that (the fork/knife/plate
        start ~1.6cm above the table, the cup slightly interpenetrating it)
        -- not called from `reset()` itself because doing so shifted the
        cup regression test's exact numbers enough to expose a separate,
        pre-existing fragility in `place()`'s release/retreat step (not
        touched here, per "preserve currently working functionality").
        `grasping/planner.py` calls this before estimating geometry for the
        elongated-object path, which has no prior behavior to preserve."""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        return self._observe()

    def step(self, ctrl: np.ndarray, n_substeps: int = 5) -> Observation:
        assert ctrl.shape == (self.model.nu,), f"expected ctrl shape ({self.model.nu},), got {ctrl.shape}"
        self.data.ctrl[:] = ctrl
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self._observe()

    def render(self, camera: str = "overhead_cam") -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def site_xpos(self, site_name: str) -> np.ndarray:
        return self.data.site(site_name).xpos.copy()

    def body_xpos(self, body_name: str) -> np.ndarray:
        return self.data.body(body_name).xpos.copy()

    def current_ctrl(self) -> np.ndarray:
        return self.data.ctrl.copy()

    def actuator_index(self, side: str, joint: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}/{joint}")

    def _observe(self) -> Observation:
        object_positions = {name: self.body_xpos(name) for name in TABLEWARE}
        gripper_positions = {side: self.site_xpos(f"{side}/gripper") for side in SIDES}
        return Observation(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            object_positions=object_positions,
            gripper_positions=gripper_positions,
        )

    def close(self):
        self._renderer.close()
