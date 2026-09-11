"""Speechmatics real-time transcription -- the bonus track's voice front-end.

Feeds the SAME planner as typed text: a transcript is just another
`instruction` string into `plan_from_instruction`. This is what makes one
repo satisfy both the Online track and the "Best Use of Speechmatics" bonus
without a second, parallel demo app.

VERIFICATION NEEDED: this implements Speechmatics' documented real-time
WebSocket protocol (StartRecognition / AddAudio / AddTranscript messages)
from training knowledge, not a tested integration -- no Speechmatics API key
was available while building this. Confirm against
https://docs.speechmatics.com/rt-api-ref before relying on it live:
specifically the auth method (this assumes a bearer API key; Speechmatics
has at times required exchanging a permanent key for a short-lived JWT via
their management API instead) and the exact real-time endpoint URL/region.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

import websockets

DEFAULT_ENDPOINT = "wss://eu2.rt.speechmatics.com/v2"
SAMPLE_RATE_HZ = 16000


async def stream_transcripts(
    audio_chunks: AsyncIterator[bytes],
    api_key: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    language: str = "en",
) -> AsyncIterator[str]:
    """Yields final transcript strings as Speechmatics reports them.

    `audio_chunks` is an async iterator of raw 16kHz mono PCM16 byte chunks
    (e.g. from a microphone capture loop) -- this module owns none of the
    audio capture itself, only the wire protocol to Speechmatics.
    """
    key = api_key or os.environ.get("SPEECHMATICS_API_KEY")
    if not key:
        raise RuntimeError("SPEECHMATICS_API_KEY not set")

    async with websockets.connect(
        endpoint, additional_headers={"Authorization": f"Bearer {key}"}
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "message": "StartRecognition",
                    "audio_format": {
                        "type": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": SAMPLE_RATE_HZ,
                    },
                    "transcription_config": {
                        "language": language,
                        "operating_point": "enhanced",
                        "max_delay": 2.0,
                    },
                }
            )
        )

        async def _pump_audio():
            async for chunk in audio_chunks:
                await ws.send(chunk)
            await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": 0}))

        pump_task = asyncio.create_task(_pump_audio())
        try:
            async for raw_message in ws:
                event = json.loads(raw_message)
                if event.get("message") == "AddTranscript":
                    transcript = event.get("metadata", {}).get("transcript", "").strip()
                    if transcript:
                        yield transcript
                elif event.get("message") == "EndOfTranscript":
                    break
        finally:
            pump_task.cancel()
