"""PUBLIC INTERFACE of the voice module: start/stop/resume a live session.

Nothing outside this module may import voice internals.

A session is the one piece of genuinely stateful, non-disposable work in the
system: it holds a live call, and if this process dies someone's interview dies
with it. Three things follow from that, and all three are visible below.

- Every turn is checkpointed to Redis, so a rejoin resumes numbering and the
  time budget instead of starting over. The invite is multi-use for exactly
  this reason.
- The recording is uploaded on the way out, in a ``finally``, because a session
  that ends badly is precisely the one whose audio matters most.
- ``stop`` is idempotent and never raises. It is called from a disconnect
  handler, a timeout, and an explicit end, and any of them can arrive twice.

Everything this module tells the rest of the system goes over ``core.events``.
It imports no models and no services other than the interview lookups it needs
to build a prompt.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from redis.asyncio import Redis

from app.core.config import settings
from app.core.events import SessionEnded, SessionStarted, publish
from app.db.session import tenant_session
from app.integrations import storage
from app.modules.interview import checkpoint, transcript
from app.modules.voice import context as context_builder
from app.modules.voice import pipeline as pipeline_builder
from app.modules.voice import prewarm
from app.modules.voice import transport as transport_builder
from app.modules.voice.observers import TranscriptObserver

log = structlog.get_logger(__name__)

# How often the session snapshots its position. Every turn would be ideal;
# every turn is also what this is, since the check runs on the turn boundary.
CHECKPOINT_EVERY_TURNS = 1

# The one end reason that does not end the interview. A reconnecting candidate
# supersedes their own previous session; see _finalise for why publishing a
# SessionEnded for it would lock them out.
SUPERSEDED = "superseded"


@dataclass
class VoiceSession:
    """One live interview. Owned by the registry below, never constructed directly."""

    interview_id: uuid.UUID
    org_id: uuid.UUID
    candidate_id: uuid.UUID
    connection: SmallWebRTCConnection
    built: pipeline_builder.BuiltPipeline
    runner: PipelineRunner
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _task: asyncio.Task | None = None
    _stopped: bool = False
    _reason: str = "completed"

    @property
    def elapsed_ms(self) -> int:
        return int((datetime.now(UTC) - self.started_at).total_seconds() * 1000)


# interview_id -> session. One live session per interview: a second connection
# for the same interview replaces the first rather than running alongside it,
# which is what a candidate reconnecting after a crash actually looks like.
_sessions: dict[uuid.UUID, VoiceSession] = {}


def active_count() -> int:
    return len(_sessions)


def is_active(interview_id: uuid.UUID) -> bool:
    return interview_id in _sessions


async def start(
    *,
    org_id: uuid.UUID,
    interview_id: uuid.UUID,
    candidate_id: uuid.UUID,
    connection: SmallWebRTCConnection,
    redis: Redis | None = None,
) -> VoiceSession:
    """Build and run a session for one interview.

    The interview is NOT transitioned here. A ``SessionStarted`` event is
    published and ``interview/service`` decides what that means for the status
    and freezes the plan -- this module never writes an interview status.
    """
    await stop(interview_id, reason=SUPERSEDED)

    # Resume position before anything else: the prompt is built from the plan,
    # and a rejoining candidate should not hear question one again.
    resume_from = 0
    elapsed_ms = 0
    if redis is not None:
        saved = await checkpoint.load(redis, interview_id)
        if saved is not None:
            resume_from = saved.next_ordinal
            elapsed_ms = saved.elapsed_ms
            log.info(
                "voice.resuming",
                interview_id=str(interview_id),
                from_ordinal=resume_from,
                elapsed_ms=elapsed_ms,
            )

    async with tenant_session(org_id, "system", None) as session:
        built_context = await context_builder.build(session, interview_id)
        # Redis may be unavailable or the checkpoint expired; the persisted
        # transcript is the fallback source of truth for numbering.
        if resume_from == 0:
            resume_from = await transcript.next_ordinal(session, interview_id)

    observer = TranscriptObserver(
        org_id=org_id,
        interview_id=interview_id,
        start_ordinal=resume_from,
    )
    transport = transport_builder.build(connection)
    built = pipeline_builder.build(
        transport=transport, messages=built_context.messages, observer=observer
    )

    # Warm Magpie before the greeting. Failure here is logged, never fatal.
    await prewarm.warm_tts()

    # Buffering is opt-in: without this the processor passes frames through and
    # records nothing, and the omission is invisible until the recording is
    # missing at the end of a real interview.
    await built.audio_buffer.start_recording()

    session_obj = VoiceSession(
        interview_id=interview_id,
        org_id=org_id,
        candidate_id=candidate_id,
        connection=connection,
        built=built,
        runner=PipelineRunner(handle_sigint=False),
    )
    # Backdate the clock so a resumed session inherits the time already spent
    # rather than getting a fresh 45 minutes.
    if elapsed_ms:
        session_obj.started_at = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - elapsed_ms / 1000, tz=UTC
        )

    _sessions[interview_id] = session_obj
    _wire_disconnect(session_obj)
    session_obj._task = asyncio.create_task(_run(session_obj, redis))

    publish(
        SessionStarted(
            org_id=org_id, interview_id=interview_id, candidate_id=candidate_id
        )
    )
    log.info(
        "voice.session_started",
        interview_id=str(interview_id),
        questions=built_context.question_count,
        resume_from=resume_from,
    )
    return session_obj


def _wire_disconnect(session: VoiceSession) -> None:
    """End the session when the peer connection drops.

    Without this, a candidate who closes their browser leaves the session
    running until the 45-minute watchdog fires -- holding an ASR stream open,
    the interview stuck IN_PROGRESS, and the recording unwritten.

    "abandoned" rather than "completed": the candidate left, and a recruiter
    reading the report should be able to tell the difference between an
    interview that finished and one that stopped.
    """

    # *args because pipecat invokes handlers with the emitting object plus
    # whatever the event carries, and that arity differs per event. A fixed
    # signature raises inside the handler, where the only symptom is a log line
    # and a session that never ends.
    async def _on_drop(*_args) -> None:
        # Scheduled rather than awaited: this fires from inside the connection's
        # own callback, and stop() waits on the session task, which is what is
        # unwinding. Awaiting here would deadlock the teardown it is triggering.
        asyncio.create_task(stop(session.interview_id, reason="abandoned"))

    for event in ("disconnected", "closed", "failed"):
        session.connection.add_event_handler(event, _on_drop)


async def _run(session: VoiceSession, redis: Redis | None) -> None:
    """Drive the pipeline to completion, then always finish cleanly."""
    watchdog = asyncio.create_task(_watchdog(session))
    checkpointer = (
        asyncio.create_task(_checkpoint_loop(session, redis)) if redis is not None else None
    )
    try:
        await session.runner.run(session.built.task)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a crash must still finalise
        session._reason = "abandoned"
        log.exception("voice.session_crashed", interview_id=str(session.interview_id))
    finally:
        for helper in (watchdog, checkpointer):
            if helper is not None:
                helper.cancel()
        await _finalise(session, redis)


async def _watchdog(session: VoiceSession) -> None:
    """End a session that overruns the cap.

    A candidate who walks away with the mic open otherwise holds a GPU-backed
    ASR stream and a paid LLM open indefinitely.
    """
    remaining_ms = max(0, checkpoint.max_interview_ms() - session.elapsed_ms)
    await asyncio.sleep(remaining_ms / 1000)
    log.info("voice.session_timed_out", interview_id=str(session.interview_id))
    await stop(session.interview_id, reason="timed_out")


async def _checkpoint_loop(session: VoiceSession, redis: Redis) -> None:
    """Snapshot position roughly once a turn.

    Polling rather than reacting to the bus, because a checkpoint must survive
    exactly the crash that loses in-flight bus events.
    """
    last_ordinal = -1
    while True:
        await asyncio.sleep(2)
        ordinal = session.built.observer.next_ordinal
        if ordinal == last_ordinal:
            continue
        last_ordinal = ordinal
        await checkpoint.save(
            redis,
            checkpoint.Checkpoint(
                interview_id=str(session.interview_id),
                next_ordinal=ordinal,
                question_ordinal=session.built.observer.question_ordinal,
                elapsed_ms=session.elapsed_ms,
            ),
        )


async def _finalise(session: VoiceSession, redis: Redis | None) -> None:
    """Flush the last turn, upload the recording, announce the end."""
    if session._stopped:
        return
    session._stopped = True
    _sessions.pop(session.interview_id, None)

    # A reply cut off mid-sentence by a disconnect still belongs in the record.
    session.built.observer.flush_bot_turn()

    recording_key = await _upload_recording(session)
    superseded = session._reason == SUPERSEDED

    if redis is not None and not superseded:
        # A superseded session is a reconnect: its checkpoint is what the
        # replacement resumes from, so it must survive.
        await checkpoint.clear(redis, session.interview_id)

    if superseded:
        # NO SessionEnded. The interview is not ending -- this session is being
        # replaced by the one that just started for the same candidate.
        #
        # Publishing it would be actively harmful: `superseded` is not a known
        # end reason, so interview/service would map it to the ABANDONED
        # default, which is terminal. The replacement session's SessionStarted
        # would then be an illegal transition, swallowed by the fire-and-forget
        # bus, and a candidate who merely reconnected would be permanently
        # locked out of the interview the multi-use invite exists to let them
        # rejoin.
        log.info("voice.session_superseded", interview_id=str(session.interview_id))
        return

    publish(
        SessionEnded(
            org_id=session.org_id,
            interview_id=session.interview_id,
            reason=session._reason,
            recording_key=recording_key,
        )
    )
    log.info(
        "voice.session_ended",
        interview_id=str(session.interview_id),
        reason=session._reason,
        recording=recording_key,
        elapsed_ms=session.elapsed_ms,
    )


async def _upload_recording(session: VoiceSession) -> str | None:
    """Persist the call audio. Never raises.

    The recording is the durable artifact the offline transcript pass and the
    confidence signals are computed from, so losing it degrades the report --
    but failing the session shutdown over it would be worse.
    """
    try:
        buffer = session.built.audio_buffer
        if not buffer.has_audio():
            return None
        audio = buffer.merge_audio_buffers()
        if not audio:
            return None
        key = f"{session.org_id}/{session.interview_id}/{uuid.uuid4().hex}.wav"
        await storage.put_bytes(
            bucket=settings.s3_bucket_recordings,
            key=key,
            data=_to_wav(audio),
            content_type="audio/wav",
        )
        return key
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "voice.recording_upload_failed",
            interview_id=str(session.interview_id),
            error=str(exc)[:200],
        )
        return None


def _to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a WAV container so the artifact is playable as-is."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(pipeline_builder.RECORDING_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(pipeline_builder.RECORDING_SAMPLE_RATE)
        handle.writeframes(pcm)
    return buffer.getvalue()


async def stop(interview_id: uuid.UUID, *, reason: str = "completed") -> bool:
    """End a session. Idempotent, and never raises.

    Returns whether there was one to stop. Called from the disconnect handler,
    the watchdog, and an explicit end -- any of which can arrive twice or for an
    interview that never had a session.
    """
    session = _sessions.get(interview_id)
    if session is None:
        return False

    session._reason = reason
    try:
        await session.built.task.cancel()
    except Exception:  # noqa: BLE001
        log.warning("voice.stop_cancel_failed", interview_id=str(interview_id))

    if session._task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(session._task), timeout=10)
        except (TimeoutError, asyncio.CancelledError):
            log.warning("voice.stop_timeout", interview_id=str(interview_id))
    return True


async def stop_all(reason: str = "abandoned") -> None:
    """Drain every live session. For graceful shutdown."""
    for interview_id in list(_sessions):
        await stop(interview_id, reason=reason)
