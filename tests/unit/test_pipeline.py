"""The shape of the post-interview chain.

Structure, not execution: no broker is involved. What these pin is that every
link stays independently re-runnable against an interview id, which is the
property that makes a half-failed pipeline recoverable by hand at 2am.
"""

import uuid

from app.workers import pipeline

ORG = uuid.uuid4()
INTERVIEW = uuid.uuid4()


def _signatures(signature) -> list:
    """Flatten a canvas into its leaf signatures, in execution order.

    Recursive because Celery rewrites what you build: everything declared after
    a chord is absorbed into that chord's body as a nested chain, so the tree is
    deeper than the ``chain(...)`` call reads.
    """
    header = getattr(signature, "tasks", None)
    body = getattr(signature, "body", None)
    if header is None and body is None:
        return [signature]

    flat = []
    for child in header or []:
        flat.extend(_signatures(child))
    if body is not None:
        flat.extend(_signatures(body))
    return flat


def test_every_link_is_an_immutable_signature():
    """``.si()``, not ``.s()``. A chain passes the previous result as the first
    positional argument, so a mutable link would silently receive a dict where
    it expects an org id -- and could not be re-run by hand at all."""
    for signature in _signatures(pipeline.build(ORG, INTERVIEW)):
        assert signature.immutable, f"{signature.task} would be fed the previous result"


def test_every_link_takes_the_same_two_ids():
    for signature in _signatures(pipeline.build(ORG, INTERVIEW)):
        assert signature.args == (str(ORG), str(INTERVIEW)), signature.task


def test_the_order_encodes_the_data_dependencies():
    """Each of these is a "step N prints or reads what step N-1 wrote"."""
    names = [s.task for s in _signatures(pipeline.build(ORG, INTERVIEW))]
    assert names[0] == "interview.finalize"

    # The scorer verifies its quotes against the transcript, so the correction
    # has to land first or every citation is checked against text about to change.
    assert names.index("scoring.correct_transcript") < names.index("scoring.score_interview")
    # The verdict is recomputed from the full event set, including what the
    # vision pass writes.
    assert names.index("proctoring.analyze_frames") < names.index("proctoring.finalize_verdict")
    # Both reports print results settled by everything above them. A recruiter
    # PDF rendered before the verdict would say "no verdict recorded" forever.
    for report in ("reports.render_recruiter", "reports.render_candidate"):
        assert names.index("scoring.score_interview") < names.index(report)
        assert names.index("proctoring.finalize_verdict") < names.index(report)


def test_the_two_reports_render_concurrently():
    """Independent work, both slow, neither reads what the other produces."""
    names = [s.task for s in _signatures(pipeline.build(ORG, INTERVIEW))]
    assert {"reports.render_recruiter", "reports.render_candidate"} == set(names[-2:])


def test_the_two_slow_independent_steps_run_concurrently():
    """Audio analysis and webcam vision touch nothing the other reads, and both
    are slow. Serialising them would add their durations rather than take the
    larger."""
    chord = pipeline.build(ORG, INTERVIEW).tasks[2]
    parallel = {t.task for t in chord.tasks}
    assert parallel == {"scoring.measure_signals", "proctoring.analyze_frames"}


def test_enqueue_swallows_a_broker_failure(monkeypatch):
    """The interview is already over and the transcript already persisted. A
    dead broker must not crash the voice session's shutdown path."""

    def _explode():
        raise ConnectionError("broker is down")

    monkeypatch.setattr(
        pipeline, "build", lambda *_: type("C", (), {"apply_async": staticmethod(_explode)})()
    )
    assert pipeline.enqueue(ORG, INTERVIEW) is None
