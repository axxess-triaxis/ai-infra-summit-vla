"""Multi-modal reasoning layer for the bimanual dinner-table task.

This is the "VLA" in the Intel Online track's framing, scoped realistically
for a hackathon: rather than training VLA model weights, a vision-capable
LLM call plays the reasoning role -- look at the rendered scene, read the
instruction (typed OR transcribed from Speechmatics), and emit a structured
pick/place plan. The low-level control/ package turns that plan into real
joint targets; this module never touches joint angles.

Provider: any OpenAI-compatible chat-completions endpoint -- Groq by
default, since Groq currently offers exactly two vision-capable models
(confirmed live against console.groq.com/docs/vision: qwen/qwen3.6-27b and
qwen/qwen3.8-27b are the only multimodal ones on their platform right now;
their Llama/GPT-OSS models are text-only). Switching to OpenRouter or any
other OpenAI-compatible provider is a matter of changing three env vars
(LLM_BASE_URL/LLM_API_KEY/LLM_MODEL), not code, since both speak the same
chat-completions shape `openai`'s SDK already implements."""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from PIL import Image

from aisummit.sim.env import SIDES, TABLEWARE

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3.6-27b"

_SYSTEM_PROMPT = f"""You are the reasoning layer for a bimanual tabletop-manipulation robot \
(two 6-DOF arms, "left" and "right", each with a parallel gripper) looking at an overhead \
camera view of a dinner table. The objects on the table are: {", ".join(TABLEWARE)}.

Given the image and a natural-language instruction, output ONLY a JSON array of steps, \
each step one of:
  {{"arm": "left"|"right", "action": "pick", "object": "<one of the table objects>"}}
  {{"arm": "left"|"right", "action": "place", "target_xyz": [x, y, z]}}

Rules:
- A "place" step must immediately follow the "pick" step for the same arm.
- target_xyz is in the robot's world frame (meters), where the table surface is z=0,
  the left arm base is at roughly (-0.47, -0.02, 0.02), the right arm base at (0.47, -0.02, 0.02),
  and reachable tabletop points are roughly x in [-0.35, 0.35], y in [-0.15, 0.30].
- Use both arms in parallel where the instruction implies it (e.g. distinct objects, no shared target).
- If the instruction is ambiguous or refers to an object not on the table, output an empty array [].
- Output nothing except the JSON array -- no prose, no markdown fences.
"""


@dataclass
class PlanStep:
    arm: str
    action: str  # "pick" | "place"
    object: str | None = None
    target_xyz: list[float] | None = None


def _encode_frame(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _extract_json_array(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in model output: {text!r}")
    return json.loads(text[start : end + 1])


def plan_from_instruction(
    image: np.ndarray,
    instruction: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> list[PlanStep]:
    """One multimodal reasoning call -> a validated list of PlanStep.

    Raises ValueError if the model's output isn't parseable JSON, or a step
    references an arm/action/object outside the known sets -- callers should
    fall back to a scripted/no-op behavior rather than execute garbage.

    Reads `LLM_API_KEY`/`LLM_MODEL`/`LLM_BASE_URL` from the environment if
    the matching argument isn't passed explicitly, falling back to
    `GROQ_API_KEY` for the key specifically (the common case of "just set
    the Groq key and go").
    """
    client = OpenAI(
        api_key=api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL", base_url),
    )
    resolved_model = os.environ.get("LLM_MODEL", model)
    response = client.chat.completions.create(
        model=resolved_model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_encode_frame(image)}"},
                    },
                    {"type": "text", "text": instruction},
                ],
            },
        ],
    )
    text = response.choices[0].message.content or ""
    raw_steps = _extract_json_array(text)

    steps = []
    for raw in raw_steps:
        arm = raw.get("arm")
        action = raw.get("action")
        if arm not in SIDES or action not in ("pick", "place"):
            raise ValueError(f"Invalid plan step from model: {raw!r}")
        if action == "pick" and raw.get("object") not in TABLEWARE:
            raise ValueError(f"Invalid pick object in plan step: {raw!r}")
        steps.append(
            PlanStep(
                arm=arm,
                action=action,
                object=raw.get("object"),
                target_xyz=raw.get("target_xyz"),
            )
        )
    return steps
