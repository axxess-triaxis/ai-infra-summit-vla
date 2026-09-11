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

## Orientation-aware grasping (`grasping/`)

The IK-is-position-only limitation above was real and has been substantially addressed --
`control/ik.py`'s `solve_ik` now takes an optional `target_quat` (omit it and it's the exact
same position-only solve it always was; every existing caller is unaffected), and a new
`grasping/` package sits above it doing real geometric reasoning instead of "find XYZ and
grab":

```
estimate_object_geometry()  ->  generate_candidates()  ->  evaluate_candidate() (scored,
   (geometry.py, reads         (candidates.py, sweeps       ranked, collision- and
   MuJoCo's own body/geom       the arm's own wrist_rotate   IK-filtered; scoring.py)
   pose -- see its docstring    joint, measures the real
   for the perception           finger-to-finger line via
   scoping trade-off)           forward kinematics)
                                        |
                                        v
                          grasp_object() executes the best-scoring
                          candidate, verifies the lift physically,
                          and retries the next-best candidate (never
                          repeating one) on failure  (planner.py)
```

- Radially symmetric objects (cup, plate) take the **exact original `pick()` path**,
  unchanged -- `grasp_object` checks `geometry.is_elongated` first and only runs the new
  machinery for the fork/knife case. `tests/test_grasping.py::test_radial_object_uses_unmodified_pick_path`
  asserts the candidate generator never even runs for the cup.
- `assets/mujoco_menagerie/aloha/dinner_table.xml` now authors a `{object}/handle` site (and
  a `{object}/unsafe` blade-region site for the knife) on the fork and knife bodies --
  ground-truth grasp-region geometry standing in for a real segmentation+PCA stage.
- `grasping/collision.py` poses a candidate's solved joint angles on a scratch `MjData` (no
  dynamics stepped) and checks MuJoCo's own contact detection against both other tableware
  objects and the table/floor.
- `grasping/debug.py` gives a structured trace of every candidate (accepted/rejected and
  why) plus `draw_grasp_overlay()`, which projects the object's principal axis, handle,
  unsafe region, and selected grasp point onto the overhead camera image using MuJoCo's own
  camera intrinsics -- verified working (see the trace/overlay evidence in the implementation
  writeup).
- `grasping/scoring.py::GraspWeights` holds every scoring weight as one configurable
  dataclass rather than literals scattered through the code.
- `grasping/planner.py::stabilize_with_other_arm` is a real, tested, opt-in bimanual
  capability (the other arm lightly touches the object's far end while the grasp arm works)
  -- not force-enabled, since the current object set's widths don't naturally cross the
  low-stability trigger.

**Six real bugs were found and fixed while getting this reliable**, each root-caused via
direct physics inspection (contact lists, joint tracking error, forward-kinematics offsets),
not guessed:

1. `collision.py` originally never checked collisions against the table/floor itself, only
   other tableware.
2. The original two-waypoint "approach then descend" pattern (copied from `pick()`) measurably
   *hurt* IK convergence for twisted, non-top-down orientations (a kinematic elbow-flip
   effect) -- `pick_oriented`/`place_oriented` now move directly to the grasp pose in one stage.
3. **The named `{side}/gripper` site IK targets is not the same point as the actual
   finger-closing midpoint** -- a fixed ~1.4cm mechanical offset (calibrated empirically,
   confirmed constant across a 120-degree wrist sweep and identical for both arms; see
   `control/ik.py::GRIPPER_SITE_TO_FINGER_MIDPOINT_OFFSET`). Invisible for the cup (3cm radius
   swallows a 1.4cm error) but larger than the fork's entire half-width -- the single biggest
   reason early grasp attempts silently missed. `finger_target_to_site_target()` converts a
   true finger-midpoint target into the site target `solve_ik` needs.
4. **Cold-start IK can fail to converge for a far, heavily-twisted target that a warm-started
   solve reaches easily.** Found via a real false positive: a candidate that scored as
   collision-free and well-aligned was actually nowhere near its intended target, because the
   check that "confirmed" it only verified collision-freedom, not position error. `solve_ik`
   gained an optional `seed_angles` (additive, same pattern as `target_quat`); `generate_candidates`
   now stores each candidate's own wrist-swept seed configuration and every downstream solve
   (scoring, execution) warm-starts from it.
5. **Authored object heights aren't resting-contact height** -- the fork/knife/plate start
   ~1.6cm above the table and settle under gravity within ~50 physics steps; geometry
   estimated immediately after `reset()` (before any arm motion) was stale by the time the arm
   actually got there. Fixed with `DinnerTableEnv.settle()`, called only from the
   elongated-object grasp path -- **not** from `reset()` itself, because doing so shifted the
   cup regression test's exact numbers enough to expose a separate, pre-existing fragility in
   `place()`'s release/retreat step. Keeping `reset()` untouched for the radial path and
   settling only where nothing previously depended on the un-settled state is what let this
   land without breaking the cup.
6. A **grasp-height clearance margin** (added to give the gripper mechanism vertical clearance
   from the table at wide wrist angles) had been implicitly tuned against the *stale* fork
   height from bug 5 -- once that was fixed, the same clearance constant needed retuning
   against the real, lower height.

## Finger-height-aware scoring

The previous iteration's "fastest next step" (penalize finger-height mismatch in scoring) was
implemented and gave real, measured improvement, but did not close the loop:

- `geometry.py::ObjectGeometry.thickness` -- the object's shortest half-extent (for a box, the
  flat dimension; for a cylinder, falls back to radius since no elongated cylinder exists in
  the current object set to calibrate against).
- `scoring.py` -- `evaluate_candidate` now measures `finger_left_z`/`finger_right_z` directly
  from the same solved-angle forward kinematics every other metric already uses, and scores
  `finger_height_mismatch` with the same clipped-linear shape already used for
  `config_distance_score`/`workspace_margin` (no new scoring idiom introduced), tolerance
  scaled to `2*thickness + 5mm` so a thin object is judged strictly and a thick one leniently.
  `GraspWeights.finger_height_symmetry` (default 2.0, same order as `orientation_alignment`)
  makes the trade-off configurable.
- `candidates.py` -- the wrist sweep went from 15-degree to 5-degree steps. This was tested
  as unnecessary at first (a 15-degree sample, `wrist=105deg`, appeared to have both good
  alignment and near-zero mismatch) -- then found to be a measurement artifact: that
  "candidate" was actually a badly non-converged solve sitting nowhere near the target (0.14m
  position error, 1.5 rad orientation error); its "good" mismatch was meaningless because the
  whole pose was wrong. Checking `ik_ok` before trusting any derived metric is what caught
  this. The finer sweep was then genuinely necessary to find real, converged options near the
  good-alignment region.

**Measured result:** at 5-degree resolution, checked across a height sweep (1.5cm-6cm
clearance) and against *both* arms, no candidate exists for the fork's current position that
is simultaneously well-aligned (closing axis near-perpendicular to the fork) and vertically
level (fingers within a few mm of each other in height) -- the practical floor for
well-aligned candidates is a consistent ~9mm finger-height mismatch, regardless of grasp
height or which arm reaches for it. That's larger than the fork's entire 8mm thickness.
Scoring correctly identifies and selects the best genuinely-available trade-off (`wrist=90deg`,
alignment 0.91, mismatch 9.3mm, score 0.734 -- versus 0.846 before this fix, appropriately
lower since the mismatch is now honestly penalized) but that trade-off still isn't good enough
to grasp an object this thin.

**Root cause, confirmed by direct measurement** (not inferred): at the selected candidate,
before closing, `left_finger` sits at z=20.8mm and `right_finger` at z=61.1mm, while the fork's
surface spans roughly z=[-1.9mm, 6.1mm]. Only the left finger is anywhere near the object.
After closing, contact logs confirm exactly that: 2 contacts recorded between the left finger
and the fork, 0 between the right finger and the fork, for the entire hold+lift sequence -- the
fork is grazed by one finger and never captured. This is failure mode **A: fingers never
contact both sides** (see the debugging checklist this iteration worked through), not a slip,
not an orientation surprise, not a friction/force issue -- a fixed ~5.5cm gap between the two
fingers' heights that no amount of retry or re-scoring within the current candidate space
closes. The cup and plate grasp reliably throughout all of this (verified after every change,
including the finer wrist sweep and both scoring changes).

## Spatial repositioning search

Next question tested: does moving the grasp POINT (not just the wrist orientation) around the
fork open a feasible bilateral grasp? Answered experimentally, not assumed.

- `geometry.py::transverse_axis_h()` -- unit horizontal vector perpendicular to the object's
  principal axis, reusing the existing axis abstraction rather than inventing a second one.
- `candidates.py` -- `generate_candidates` now varies the grasp point across a small star
  pattern (each offset axis independently around the base point, not a full grid): 3
  longitudinal offsets (unchanged) + 2 new transverse offsets = 5 spatial points, each still
  swept across the full 37-angle wrist range (185 candidates total, not the "huge brute-force
  grid" the brief explicitly ruled out). World-frame X/Y offsets aren't a separate dimension:
  the fork/knife have zero rotation, so their principal/transverse axes already ARE world Y/X --
  a separate world-frame sweep would just repeat the same numbers under different labels. The
  mechanism itself is general (object-relative axes), not fork-specific.
- `scoring.py` -- feasibility is now an explicit, separate classification from score
  (`GraspMetrics.feasible`/`infeasible_reason`), not just a low weighted number. A new
  `_finger_reaches_object()` check catches something the mismatch-only metric structurally
  cannot: two fingers can be perfectly symmetric with each other while BOTH floating equally
  far from the object. `evaluate_candidate` now checks, in order, `ik_ok` -> `collision_valid`
  -> each finger individually within `thickness + 5mm` of the object's own height -> mismatch
  within tolerance -- any failure makes the candidate infeasible outright, regardless of how
  high its alignment score is.
- `planner.py` -- `grasp_object`'s filtering loop now gates on `metrics.feasible` directly
  (folding what used to be two separate `collision_valid`/`ik_ok` checks into one, richer
  classification) instead of scoring a physically-impossible candidate down and hoping it loses.
- `debug.py` -- new `GraspDebugTrace.spatial_search_summary()`: candidates evaluated,
  IK-converged, geometrically feasible, and the best of each of the feasible/infeasible groups.

**This stricter check surfaced a second, independent problem**, on top of the wrist-tilt
coupling from the previous iteration: `_GRASP_HEIGHT_CLEARANCE` (added earlier so the gripper
mechanism clears the table at wide wrist angles) is applied to the literal close point, not
just an approach waypoint -- `pick_oriented` is deliberately single-stage (see its docstring:
a two-waypoint version caused a worse kinematic elbow-flip), so there is no separate descent
back down to the object's real height before closing. Measured directly: the nominal target
sits ~30mm above the fork's actual surface, for *every* candidate, regardless of wrist angle or
spatial offset. Lowering the clearance to compensate was also tried and measured: at any
clearance low enough to bring the fingers within reach, the well-aligned wrist angles go back
to penetrating the table -- the exact problem the clearance was added to fix in the first
place. This is a genuine height-vs-clearance conflict, independent of horizontal position.

**Experimental answer to the assigned question:** across all 5 spatial offsets, swept against
6 clearance heights (12mm-30mm) and a fine wrist sweep (50-130 degrees in 5-degree steps) --
30 x 17 = 510 combinations per spatial point, 2,550 total -- zero candidates are simultaneously
collision-free, within IK tolerance, well-aligned (>0.5), and within finger-reach of the fork.
**Moving the grasp point horizontally does not open a feasible bilateral grasp window at the
fork's current pose.** The production search (default clearance, no manual override) reports
the same result cleanly: `candidates evaluated: 185, IK-converged: 94 (left) / 185 (right),
geometrically feasible: 0` for both arms, best infeasible candidate `align=0.97, mismatch=9.3mm,
reason="neither finger reaches the object"`.

No physical APPROACH->CLOSE->HOLD->LIFT run was performed this iteration: the brief is explicit
that faking a success when no feasible candidate exists is worse than reporting the negative
result, and every avenue this pipeline currently searches (wrist orientation, grasp height,
horizontal position, both arms) converges on the same conclusion.

**Next geometric intervention** (not attempted here -- out of this iteration's scope, which was
horizontal position only): decouple "approach clearance for table avoidance" from "final close
height" in `pick_oriented`, most plausibly via a controlled *translation-only* final descent
after the orientation is already locked in (translation-only should not reintroduce the
elbow-flip a full 6-DOF two-waypoint solve caused previously, since the hard part --
orientation -- would already be solved before descending). This is a real code change to
`control/primitives.py`, not a scoring or search-space tweak, which is why it wasn't done as
part of a "search object/approach position" task.

- **Speechmatics integration is unverified.** Implemented from documented protocol knowledge,
  not tested against a live key. Confirm the auth flow (bearer key vs. short-lived JWT
  exchange) and endpoint region before the actual demo.
- **Object set is four primitives**, not photorealistic tableware meshes -- fine for a
  reasoning/manipulation demo, not for a visual polish pass.
