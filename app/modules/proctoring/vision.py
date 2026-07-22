"""Webcam frame analysis: face absent, multiple faces, gaze off-screen.

Runs OFFLINE, in a worker, after the interview. Never on the turn budget: a
1.6-second VLM call per frame would be catastrophic mid-conversation, and there
is nothing a live analysis could do that an after-the-fact one cannot, because
the product does not interrupt candidates on a machine's say-so.

The model is asked for counts, not judgements. "How many faces are visible" is
something a VLM answers reliably; "is this person cheating" is not a question it
should be asked, and a model that answered it would be laundering a guess into
what looks like a finding.

Verified against nvidia/nemotron-nano-12b-v2-vl: a synthetic two-face frame came
back as {"faces": 2} in ~1.6s.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from app.integrations import nim_client
from app.models.proctoring import ProctorEventType, ProctorSeverity
from app.modules.voice.nvidia.catalog import get_service

log = structlog.get_logger(__name__)

PROMPT = (
    "You are reviewing a webcam still from a remote job interview. Report only "
    "what is visibly present -- do not speculate about intent or behaviour.\n\n"
    'Reply with ONLY this JSON: {"faces": <int>, "looking_at_screen": <bool>, '
    '"screens_visible": <int>, "note": "<at most 12 words>"}'
)


class FrameAnalysis(BaseModel):
    """What the model reports about one still."""

    faces: int = Field(ge=0, le=20)
    looking_at_screen: bool = True
    # A second monitor is not misconduct by itself; a reviewer may want to know.
    screens_visible: int = Field(default=0, ge=0, le=10)
    note: str = Field(default="", max_length=200)


@dataclass(frozen=True, slots=True)
class FrameFinding:
    """A frame turned into zero or more proctoring signals."""

    event_type: ProctorEventType
    severity: ProctorSeverity
    note: str


async def analyse_frame(image_bytes: bytes, *, content_type: str = "image/jpeg") -> FrameAnalysis:
    """One still through the VLM. Raises NimError if the model will not answer."""
    encoded = base64.b64encode(image_bytes).decode()
    messages = [
        {
            "role": "user",
            "content": [  # type: ignore[dict-item]
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{encoded}"},
                },
            ],
        }
    ]
    return await nim_client.complete_structured(
        messages,  # type: ignore[arg-type]
        FrameAnalysis,
        spec=get_service("vision"),
        max_tokens=200,
        # Deterministic: the same frame should not produce a different finding
        # on a re-run, or the verdict stops being defensible.
        temperature=0.0,
    )


def findings_for(analysis: FrameAnalysis) -> list[FrameFinding]:
    """Turn one analysis into signals worth recording.

    A frame where everything is normal produces nothing. Recording "one face
    present" thousands of times would bury the frames that matter.
    """
    findings: list[FrameFinding] = []

    if analysis.faces == 0:
        # WARN, not CRITICAL: people reach for water, answer a doorbell, or sit
        # slightly outside a badly-aimed webcam.
        findings.append(
            FrameFinding(
                ProctorEventType.FACE_ABSENT,
                ProctorSeverity.WARN,
                analysis.note or "no face visible",
            )
        )
    elif analysis.faces > 1:
        # No innocent reading a recruiter should not see for themselves.
        findings.append(
            FrameFinding(
                ProctorEventType.MULTIPLE_FACES,
                ProctorSeverity.CRITICAL,
                analysis.note or f"{analysis.faces} faces visible",
            )
        )

    return findings
