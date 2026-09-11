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

from aisummit.control.primitives import pick, place
from aisummit.planner.vla_planner import PlanStep, plan_from_instruction
from aisummit.sim.env import DinnerTableEnv
from aisummit.voice.speechmatics_client import stream_transcripts

OUTPUT_DIR = Path("outputs")


def execute_plan(env: DinnerTableEnv, steps: list[PlanStep]):
    ctrl = env.current_ctrl()
    obs = None
    for step in steps:
        if step.action == "pick":
            print(f"  {step.arm} arm: pick {step.object}")
            ctrl, obs = pick(env, ctrl, side=step.arm, object_name=step.object)
        elif step.action == "place":
            print(f"  {step.arm} arm: place at {step.target_xyz}")
            ctrl, obs = place(env, ctrl, side=step.arm, target_xyz=step.target_xyz)
    return obs


def run(instruction: str):
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
        execute_plan(env, steps)

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
    transcript = None
    async for text in stream_transcripts(_wav_chunks(wav_file)):
        print(f"  transcript: {text}")
        transcript = text
    if transcript is None:
        print("No transcript produced.")
        return
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
