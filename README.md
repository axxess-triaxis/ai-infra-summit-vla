# AI Infra Summit 2026 -- Bimanual VLA + Speechmatics

One repo covering two tracks from the AI Infra Summit "Choose Your Path" email:

- **Online track (Intel -- Bimanual VLA Manipulation with Multi-Modal Reasoning)**: two
  simulated arms set a dinner table (plate, fork, knife, cup), simulation-only, no hardware.
- **Bonus track (Best Use of Speechmatics)**: the same task driven by a spoken instruction
  instead of typed text.

They share one code path on purpose: a Speechmatics transcript and a typed instruction both
just become the `instruction` string handed to the same planner. There is no separate voice
demo bolted on the side.

## Architecture

```
mic / WAV  --Speechmatics-->  transcript  --\
                                              >--  plan_from_instruction()  -->  [PlanStep, ...]
typed text  ------------------------------- /              |
                                                             v
                                              control/primitives.py (pick/place)
                                                             |
                                                             v
                                              control/ik.py (damped least-squares)
                                                             |
                                                             v
                                              sim/env.py  (mujoco, ALOHA bimanual rig)
```

- **`sim/env.py`** -- `DinnerTableEnv`, a thin wrapper directly on the official `mujoco` Python
  bindings (not robosuite -- see "Why not robosuite" below). Loads
  `assets/mujoco_menagerie/aloha/dinner_table.xml`: Google DeepMind's ALOHA two-arm rig
  (2x 6-DOF ViperX 300 arms + parallel grippers, position-controlled) plus four tableware
  bodies we added (plate, fork, knife, cup).
- **`control/ik.py`** -- position-only damped least-squares IK against each arm's
  `{side}/gripper` site. No orientation term (see Known limitations).
- **`control/primitives.py`** -- `pick(env, ctrl, side, object_name)` and
  `place(env, ctrl, side, target_xyz)`, built entirely on the IK solver plus the arms'
  native position actuators. This is the only thing the planner's output ever drives --
  it never sees a joint angle.
- **`planner/vla_planner.py`** -- the "VLA" reasoning layer. Sends the overhead camera
  render + the instruction to Claude (`claude-sonnet-5`, vision-capable) with a system
  prompt describing the two arms and the four objects, gets back a structured JSON plan.
  Training real VLA weights in a one-week hackathon isn't realistic; a multimodal LLM doing
  the "what to do" reasoning while `control/` does deterministic "how to move it" is the
  scoped-realistic version of the same idea.
- **`voice/speechmatics_client.py`** -- real-time transcription over Speechmatics'
  WebSocket protocol. **Not yet tested against a real Speechmatics key** (none was available
  while building this) -- see the module docstring for exactly what to verify against
  `docs.speechmatics.com` before the actual demo.
- **`demo.py`** -- the single entrypoint, `--input text` or `--input voice`.

## Setup

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY and SPEECHMATICS_API_KEY
```

The ALOHA model assets (`assets/mujoco_menagerie/aloha/`, ~20MB) are already vendored in
this repo (sparse-checked-out from `google-deepmind/mujoco_menagerie`, Apache-2.0) -- no
extra download step.

## Running it

```bash
uv run python -m aisummit.demo --input text --instruction "pick up the cup and move it to the left side of the table"
```

Writes `outputs/before.png` and `outputs/after.png`.

```bash
uv run python -m aisummit.demo --input voice --wav-file sample.wav   # 16kHz mono PCM16 WAV
```

## Tests

```bash
uv run pytest tests/ -v
```

`tests/test_pick_and_place.py` is a real, physics-simulated pick-and-place (no mocks): resets
the env, grasps the cup with the right arm, moves it, places it, and asserts the final position
and that it was actually lifted off the table mid-motion. Passing as of this commit.

## Why not robosuite

robosuite is the obvious first choice for bimanual manipulation, but it's currently broken on
native Windows in a way that isn't a config issue:

1. It ships expecting a `mujoco.dll` copy inside its own package that pip doesn't actually
   install (needs a manual copy from the `mujoco` package's own install location).
2. Once that's fixed, its offscreen renderer hardcodes `MUJOCO_GL=egl` (Linux/GPU-server-only)
   unless a private macro file disables GPU rendering -- another manual fix.
3. Even after both of those, its `OperationalSpaceController` calls `mujoco.mj_fullM(...,
   data.qM)`, and `qM` was removed from `MjData` in mujoco 3.11+ (confirmed by bisecting
   versions 3.1.6 through 3.13.0: present through 3.10.0, gone from 3.11.0 on). Pinning
   mujoco back to 3.10.0 to keep `qM` around then hits a second, deeper `mj_fullM` argument
   binding mismatch that wasn't worth chasing further.

Given a one-week window, we dropped robosuite and built directly on the official `mujoco`
bindings plus DeepMind's own ALOHA model from `mujoco_menagerie` -- verified working
end-to-end on this machine (model loads, renders, and the pick-and-place test passes).

## Known limitations (stated plainly, not hidden)

- **IK is position-only.** The fork and knife are thin, elongated boxes; grasping them
  reliably needs the gripper's approach orientation aligned to the object's long axis, which
  our IK doesn't control. The cup and plate (radially symmetric) grasp reliably from any
  approach angle -- confirmed by test. Fixing this means extending `ik.py` to a full 6-DOF
  position+orientation damped least-squares solve.
- **Speechmatics integration is unverified.** Implemented from documented protocol knowledge,
  not tested against a live key. Confirm the auth flow (bearer key vs. short-lived JWT
  exchange) and endpoint region before the actual demo.
- **Object set is four primitives**, not photorealistic tableware meshes -- fine for a
  reasoning/manipulation demo, not for a visual polish pass.
