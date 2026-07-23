"""The frame-to-event bridge, and the prompt the live model is given.

These are the two places in the voice module where a mistake is silent. A
mis-assembled observer produces a transcript that is subtly wrong -- one reply
shredded across twelve lines, or the last sentence of the interview missing --
and nothing errors. A prompt that leaks the plan produces a perfectly normal
interview in which the candidate was handed the questions.
"""

import uuid

import pytest
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

from app.core.events import EventBus, TurnCompleted
from app.modules import prompts
from app.modules.voice.observers import TranscriptObserver

ORG = uuid.uuid4()
INTERVIEW = uuid.uuid4()


@pytest.fixture
def bus(monkeypatch):
    """A private bus, with the observer's publish redirected onto it."""
    local = EventBus()
    monkeypatch.setattr("app.modules.voice.observers.publish", local.publish)
    return local


@pytest.fixture
def observer():
    return TranscriptObserver(org_id=ORG, interview_id=INTERVIEW)


def _push(frame):
    return FramePushed(
        source=None, destination=None, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0
    )


async def _feed(observer, bus, *frames):
    for frame in frames:
        await observer.on_push_frame(_push(frame))
    await bus.drain()


def _turns(bus_events: list[TurnCompleted]) -> list[tuple[str, str]]:
    return [(e.speaker, e.content) for e in bus_events]


@pytest.fixture
def captured(bus):
    events: list[TurnCompleted] = []
    bus.subscribe(TurnCompleted, lambda e: events.append(e))
    return events


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="candidate", timestamp="")


def _tts(text: str) -> TTSTextFrame:
    # aggregated_by is required and records which aggregator produced the
    # fragment; it is irrelevant to this observer, which only reads .text.
    return TTSTextFrame(text=text, aggregated_by="sentence")


# --- Candidate speech -------------------------------------------------------


async def test_a_candidate_utterance_becomes_one_turn(observer, bus, captured):
    await _feed(observer, bus, _transcription("We moved from RabbitMQ to Kafka."))
    assert _turns(captured) == [("candidate", "We moved from RabbitMQ to Kafka.")]


async def test_interim_transcriptions_are_ignored(observer, bus, captured):
    """Interim results are for display. Recording them would write a transcript
    of every half-formed guess the ASR made on the way to the real answer."""
    await _feed(
        observer,
        bus,
        InterimTranscriptionFrame(text="We moved from Rabbit", user_id="c", timestamp=""),
        InterimTranscriptionFrame(text="We moved from RabbitMQ to", user_id="c", timestamp=""),
        _transcription("We moved from RabbitMQ to Kafka."),
    )
    assert len(captured) == 1


async def test_an_empty_utterance_is_dropped(observer, bus, captured):
    await _feed(observer, bus, _transcription("   "))
    assert captured == []


# --- Interviewer speech -----------------------------------------------------


async def test_tts_fragments_are_joined_into_one_turn(observer, bus, captured):
    """The whole reason this observer buffers: TTS emits per sentence, and one
    event per fragment would shred a single reply across the transcript."""
    await _feed(
        observer,
        bus,
        _tts("Thanks for that. "),
        _tts("Tell me about the ledger. "),
        _tts("What was the hardest part?"),
        BotStoppedSpeakingFrame(),
    )
    assert _turns(captured) == [
        ("interviewer", "Thanks for that. Tell me about the ledger. What was the hardest part?")
    ]


async def test_an_unflushed_reply_is_not_emitted_early(observer, bus, captured):
    await _feed(observer, bus, _tts("Still speaking"))
    assert captured == []


async def test_a_reply_cut_off_by_a_disconnect_still_reaches_the_transcript(
    observer, bus, captured
):
    """No BotStoppedSpeakingFrame arrives when the call drops mid-sentence.
    The session calls flush explicitly on the way out."""
    await _feed(observer, bus, _tts("Half a sentence"))
    observer.flush_bot_turn()
    await bus.drain()

    assert _turns(captured) == [("interviewer", "Half a sentence")]


async def test_flushing_twice_does_not_duplicate(observer, bus, captured):
    await _feed(observer, bus, _tts("A reply"), BotStoppedSpeakingFrame())
    observer.flush_bot_turn()
    await bus.drain()
    assert len(captured) == 1


# --- Ordering and offsets ---------------------------------------------------


async def test_turns_are_numbered_in_conversation_order(observer, bus, captured):
    await _feed(
        observer,
        bus,
        _tts("First question?"),
        BotStoppedSpeakingFrame(),
        _transcription("An answer."),
        _tts("Second question?"),
        BotStoppedSpeakingFrame(),
    )
    assert [e.ordinal for e in captured] == [0, 1, 2]
    assert [e.speaker for e in captured] == ["interviewer", "candidate", "interviewer"]


async def test_a_resumed_session_continues_numbering():
    """A reconnecting candidate must not overwrite the turns already stored."""
    resumed = TranscriptObserver(org_id=ORG, interview_id=INTERVIEW, start_ordinal=7)
    assert resumed.next_ordinal == 7


async def test_offsets_are_relative_and_ordered(observer, bus, captured):
    await _feed(observer, bus, _transcription("An answer."))
    event = captured[0]
    # Relative to session start, so the first turn is near zero rather than at
    # a wall-clock epoch.
    assert event.started_offset_ms < 5_000
    assert event.ended_offset_ms >= event.started_offset_ms


async def test_question_ordinal_advances_once_per_interviewer_turn(observer, bus, captured):
    await _feed(observer, bus, _tts("Q1?"), BotStoppedSpeakingFrame())
    await _feed(observer, bus, _transcription("A1."))
    await _feed(observer, bus, _tts("Q2?"), BotStoppedSpeakingFrame())

    # The answer is attributed to the question that preceded it.
    assert captured[1].speaker == "candidate"
    assert captured[1].question_ordinal == 1


# --- The interviewer prompt -------------------------------------------------


def _prompt(**overrides) -> str:
    values = {
        "job_title": "Staff Engineer",
        "job_description": "Own the ledger.",
        "resume_context": "[experience] Kafka migration",
        "questions": "1. Tell me about the migration.",
        "duration_minutes": 30,
    }
    messages = prompts.render("interviewer", **{**values, **overrides})
    return "\n".join(m["content"] for m in messages)


def test_the_prompt_forbids_revealing_the_plan_or_rubric():
    text = _prompt().lower()
    assert "never reveal" in text
    assert "rubric" in text


def test_the_prompt_constrains_length_because_output_is_spoken():
    """A model writing for a screen produces bullet lists and 200-word
    paragraphs, which through TTS is an unlistenable monologue."""
    text = _prompt().lower()
    assert "one or two sentences" in text
    assert "no markdown" in text


def test_the_prompt_labels_the_resume_as_candidate_written():
    assert "written by the candidate" in _prompt().lower()


def test_the_prompt_excludes_protected_characteristics():
    text = _prompt().lower()
    for attribute in ("age", "race", "religion", "gender", "disability"):
        assert attribute in text, f"{attribute} is not named in the prohibition"


# --- The bus's own failure path ---------------------------------------------


async def test_a_failing_handler_is_logged_rather_than_crashing_the_logger():
    """The line that reports a broken subscriber must not itself break.

    OBSERVED in production logs: a handler raised, and the ``log.exception``
    meant to report it raised ``TypeError: exception() got multiple values for
    argument 'event'`` -- structlog takes the message as a positional named
    ``event``, and the call passed ``event=`` as a keyword. The original error
    vanished, and the only channel for surfacing a bad subscriber was the one
    that was broken.
    """
    import uuid

    import structlog.testing

    from app.core.events import EventBus, SessionStarted

    bus = EventBus()
    org, interview = uuid.uuid4(), uuid.uuid4()

    async def explodes(_event) -> None:
        raise RuntimeError("subscriber is broken")

    bus.subscribe(SessionStarted, explodes)

    # Captured, because "drain() returned" is not an assertion: the TypeError
    # dies inside a fire-and-forget task and is swallowed, so the test passed
    # with the bug present. What has to be true is that the record was actually
    # EMITTED.
    with structlog.testing.capture_logs() as captured:
        bus.publish(SessionStarted(org_id=org, interview_id=interview))
        await bus.drain()

    failures = [c for c in captured if c.get("event") == "event_handler_failed"]
    assert len(failures) == 1, f"the failure was never logged: {captured}"
    assert failures[0]["event_type"] == "SessionStarted"
    assert "explodes" in failures[0]["handler"]


async def test_one_broken_handler_does_not_stop_the_others():
    import uuid

    from app.core.events import EventBus, SessionStarted

    bus = EventBus()
    seen: list[str] = []

    async def explodes(_event) -> None:
        raise RuntimeError("broken")

    async def works(_event) -> None:
        seen.append("ran")

    bus.subscribe(SessionStarted, explodes)
    bus.subscribe(SessionStarted, works)
    bus.publish(SessionStarted(org_id=uuid.uuid4(), interview_id=uuid.uuid4()))
    await bus.drain()

    assert seen == ["ran"], "a broken subscriber suppressed a working one"


# --- One frame, many hops -----------------------------------------------------


async def test_a_frame_seen_at_every_hop_is_recorded_once():
    """The observer fires once per pipeline HOP, not once per frame.

    A TTSTextFrame is pushed TTS -> transport.output() -> audio_buffer ->
    assistant aggregator, so the same object arrives three times.

    OBSERVED in a real interview transcript: every interviewer line stored as
    "Hi Prateek...Hi Prateek...Hi Prateek..." -- three identical copies
    concatenated, which is also what the scorer then read.
    """
    import uuid

    from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSTextFrame

    from app.core.events import EventBus, TurnCompleted
    from app.modules.voice.observers import TranscriptObserver

    bus = EventBus()
    seen: list[TurnCompleted] = []
    bus.subscribe(TurnCompleted, lambda e: seen.append(e))

    observer = TranscriptObserver(org_id=uuid.uuid4(), interview_id=uuid.uuid4())

    import app.modules.voice.observers as observers_module

    original = observers_module.publish
    observers_module.publish = bus.publish
    try:
        frame = TTSTextFrame(text="Hello Prateek.", aggregated_by="sentence")
        # The same object, pushed at three different links.
        for _ in range(3):
            await observer.on_push_frame(_pushed(frame))
        await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))
    finally:
        observers_module.publish = original

    await bus.drain()
    assert len(seen) == 1
    assert seen[0].content == "Hello Prateek.", f"text was repeated: {seen[0].content!r}"


async def test_distinct_frames_are_all_recorded():
    """The dedupe must not swallow genuinely different fragments."""
    import uuid

    from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSTextFrame

    from app.core.events import EventBus, TurnCompleted
    from app.modules.voice.observers import TranscriptObserver

    bus = EventBus()
    seen: list[TurnCompleted] = []
    bus.subscribe(TurnCompleted, lambda e: seen.append(e))

    observer = TranscriptObserver(org_id=uuid.uuid4(), interview_id=uuid.uuid4())

    import app.modules.voice.observers as observers_module

    original = observers_module.publish
    observers_module.publish = bus.publish
    try:
        for text in ("Hello. ", "Tell me about sharding. "):
            await observer.on_push_frame(
                _pushed(TTSTextFrame(text=text, aggregated_by="sentence"))
            )
        await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))
    finally:
        observers_module.publish = original

    await bus.drain()
    assert len(seen) == 1
    assert seen[0].content == "Hello. Tell me about sharding."


def _pushed(frame):
    """Minimal stand-in for pipecat's FramePushed."""
    from types import SimpleNamespace

    return SimpleNamespace(frame=frame, source=None, destination=None, direction=None)
