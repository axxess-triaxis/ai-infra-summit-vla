"""Single entrypoint for both tracks.

    uv run python -m aisummit.demo --input text --instruction "put the cup where the plate is"
    uv run python -m aisummit.demo --input voice --wav-file sample.wav

Both paths converge on the same planner + executor: a spoken instruction is
just a transcript fed into `plan_from_instruction`, exactly like typed text.
That convergence is what lets the Online track and the Speechmatics bonus
track live in one repo instead of two demos.
"""

from __future__ import annotations

import argparse
import asyncio
import wave
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from aisummit.control.primitives import place, place_oriented
from aisummit.grasping.debug import GraspDebugTrace, draw_grasp_overlay
from aisummit.grasping.planner import acquire_object
from aisummit.planner.vla_planner import PlanStep, plan_from_instruction
from aisummit.sim.env import DinnerTableEnv
from aisummit.voice.speechmatics_client import stream_transcripts

OUTPUT_DIR = Path("outputs")


def execute_plan(env: DinnerTableEnv, steps: list[PlanStep], debug: bool = False):
    """Routes "pick" through the orientation-aware grasp pipeline
    (grasping/planner.py) -- which itself falls back to the original
    position-only pick() unchanged for radially symmetric objects -- and
    remembers each arm's achieved grasp orientation so the matching
    "place" keeps the object at a predictable orientation through
    transport (important for a knife) instead of reverting to a
    default top-down placement."""
    ctrl = env.current_ctrl()
    obs = None
    held_quat: dict[str, object] = {}
    for step in steps:
        if step.action == "pick":
            print(f"  {step.arm} arm: pick {step.object}")
            pre_grasp_frame = env.render()
            trace = GraspDebugTrace(object_name=step.object, verbose=debug)
            result = acquire_object(env, ctrl, side=step.arm, object_name=step.object, debug=trace)
            ctrl, obs = result.final_ctrl, obs
            held_quat[step.arm] = result.achieved_quat
            print(f"    grasp {'succeeded' if result.success else 'FAILED'}"
                  f" (strategy={result.strategy}, {len(trace.attempts)} attempt(s) tried)")
            if debug:
                print(trace.summary())
            if trace.geometry is not None and trace.geometry.is_elongated:
                overlay = draw_grasp_overlay(
                    pre_grasp_frame, env.model, env.data, "overhead_cam", trace.geometry,
                    selected_position=trace.geometry.grasp_region,
                )
                Image.fromarray(overlay).save(OUTPUT_DIR / f"grasp_debug_{step.object}.png")
                print(f"    saved grasp overlay to {OUTPUT_DIR / f'grasp_debug_{step.object}.png'}")
        elif step.action == "place":
            print(f"  {step.arm} arm: place at {step.target_xyz}")
            quat = held_quat.get(step.arm)
            if quat is not None:
                ctrl, obs = place_oriented(env, ctrl, side=step.arm, target_xyz=step.target_xyz, target_quat=quat)
            else:
                ctrl, obs = place(env, ctrl, side=step.arm, target_xyz=step.target_xyz)
    return obs


def run(instruction: str, debug: bool = False):
    OUTPUT_DIR.mkdir(exist_ok=True)
    env = DinnerTableEnv()
    try:
        env.reset()
        before = env.render()
        Image.fromarray(before).save(OUTPUT_DIR / "before.png")

        print(f"Instruction: {instruction!r}")
        steps = plan_from_instruction(before, instruction)
        if not steps:
            print("Planner returned no steps (ambiguous instruction, or nothing to do).")
            return
        print(f"Plan ({len(steps)} steps):")
        execute_plan(env, steps, debug=debug)

        after = env.render()
        Image.fromarray(after).save(OUTPUT_DIR / "after.png")
        print(f"Saved {OUTPUT_DIR / 'before.png'} and {OUTPUT_DIR / 'after.png'}")
    finally:
        env.close()


async def _wav_chunks(path: Path, chunk_frames: int = 3200):
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000, "expected 16kHz PCM16 mono WAV"
        assert wav_file.getsampwidth() == 2, "expected 16-bit PCM WAV"
        while chunk := wav_file.readframes(chunk_frames):
            yield chunk


async def run_voice(wav_file: Path):
    # stream_transcripts yields one final transcript per recognized segment,
    # not the whole utterance at once -- confirmed against real speech
    # ("pick up the cup and move it closer to the plate" came back as 5
    # separate chunks). Keeping only the last one (an earlier version of
    # this function did) would have silently dropped most of any real
    # instruction; every chunk has to be joined.
    chunks = []
    async for text in stream_transcripts(_wav_chunks(wav_file)):
        print(f"  transcript chunk: {text}")
        chunks.append(text)
    if not chunks:
        print("No transcript produced.")
        return
    transcript = " ".join(chunks)
    print(f"  full transcript: {transcript!r}")
    run(transcript)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=["text", "voice"], default="text")
    parser.add_argument("--instruction", default="pick up the cup and move it to the left side of the table")
    parser.add_argument("--wav-file", type=Path, default=None)
    args = parser.parse_args()

    load_dotenv()

    if args.input == "text":
        run(args.instruction)
    else:
        if args.wav_file is None:
            parser.error("--input voice requires --wav-file (16kHz mono PCM16 WAV)")
        asyncio.run(run_voice(args.wav_file))


if __name__ == "__main__":
    main()
