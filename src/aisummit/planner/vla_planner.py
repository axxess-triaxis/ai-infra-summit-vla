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
import re
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from PIL import Image

from aisummit.sim.env import SIDES, TABLEWARE

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3.6-27b"
# Groq's free tier caps this model's output at 1000 tokens/minute (hit
# live: a 429 at max_tokens=1024, just over). The expected response here
# is a short JSON array (a handful of pick/place steps), so 800 leaves
# real margin without constraining genuine plans.
_MAX_TOKENS = 800

_SYSTEM_PROMPT = f"""You are the reasoning layer for a bimanual tabletop-manipulation robot \
(two 6-DOF arms, "left" and "right", each with a parallel gripper) looking at an overhead \
camera view of a dinner table. The objects on the table are: {", ".join(TABLEWARE)}.

Given the image and a natural-language instruction, output ONLY a JSON array of steps, \
each step one of:
  {{"arm": "left"|"right", "action": "pick", "object": "<one of the table objects>"}}
  {{"arm": "left"|"right", "action": "place", "target_xyz": [x, y, z]}}

Rules:
- A "place" step must immediately follow the "pick" step for the same arm -- the SAME arm
  that picks an object is the one that must place it, there is no handoff between arms.
- target_xyz is in the robot's world frame (meters), where the table surface is z=0,
  the left arm base is at roughly (-0.47, -0.02, 0.02), the right arm base at (0.47, -0.02, 0.02).
- Each arm can only reach reliably within about 0.5m of its OWN base. Since the same arm must
  both pick and place, never target a placement more than ~0.5m from that arm's base -- if the
  instruction asks to move something toward the far/opposite side of the table, place it as far
  in that direction as that arm can actually reach (a modest shift), not the extreme opposite
  edge, since the arm doing the picking is also the one that has to reach the place target.
- Use both arms in parallel where the instruction implies it (e.g. distinct objects, no shared target).
- If the instruction is ambiguous or refers to an object not on the table, output an empty array [].
- Output nothing except the JSON array -- no prose, no markdown fences.

/no_think"""
# Trailing /no_think is Qwen3's documented convention for skipping its
# extended-thinking mode. Needed in practice, not just in theory: without
# it, qwen3.6-27b spent its whole token budget on a <think>...</think>
# block and got cut off before ever emitting JSON (found by printing the
# raw response against a live key). Paired with `reasoning_effort: "none"`
# in the actual API call below -- belt and suspenders, since only the
# combination was confirmed to reliably suppress it.


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


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def _extract_json_array(text: str) -> list[dict]:
    """Parses the model's plan output, tolerating a real failure mode seen
    live: qwen3.6-27b sometimes emits bare comma-separated objects
    (`{...},\n{...}`) instead of wrapping them in `[...]` as instructed --
    a small-model prompt-following slip, not a rare edge case worth
    treating as a hard error. Tries a real `[...]` array first; falls back
    to collecting every top-level `{...}` object (steps have no nested
    braces, so a non-greedy regex is a safe, dependency-free stand-in for
    a full parser here) and treating that list as the plan."""
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    objects = _JSON_OBJECT_RE.findall(text)
    if not objects:
        raise ValueError(f"No JSON array or objects found in model output: {text!r}")
    return [json.loads(obj) for obj in objects]


# Small models occasionally drop a required field or otherwise emit a
# malformed step even with the parsing fixes above -- confirmed live: one
# real call returned {"action": "pick"} with no "arm" key at all, while an
# identical call moments later returned a perfectly-formed plan. That's
# ordinary LLM run-to-run variance, not a parsing bug, and the standard
# mitigation is a retry, not a more elaborate parser chasing every
# possible malformation.
_MAX_ATTEMPTS = 3


def _one_attempt(client: OpenAI, model: str, image: np.ndarray, instruction: str) -> list[PlanStep]:
    response = client.chat.completions.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        extra_body={"reasoning_effort": "none"},
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


def plan_from_instruction(
    image: np.ndarray,
    instruction: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> list[PlanStep]:
    """One multimodal reasoning call -> a validated list of PlanStep,
    retried up to `_MAX_ATTEMPTS` times on a malformed response.

    Raises the last ValueError if every attempt's output isn't parseable
    JSON, or references an arm/action/object outside the known sets --
    callers should fall back to a scripted/no-op behavior rather than
    execute garbage.

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

    last_error: Exception | None = None
    for _ in range(_MAX_ATTEMPTS):
        try:
            return _one_attempt(client, resolved_model, image, instruction)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
    raise last_error
