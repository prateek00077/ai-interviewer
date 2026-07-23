"""Outbound email: when it sends, when it does not, and what it must not say.

Every caller is doing something else that has already succeeded, so the
governing property here is that nothing in this module can raise into them.
"""

import pytest

from app.integrations import email


@pytest.fixture
def smtp_configured(monkeypatch):
    monkeypatch.setattr(email.settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(email.settings, "smtp_user", "")


# --- Sending ----------------------------------------------------------------


async def test_no_smtp_host_is_a_no_op_not_an_error(monkeypatch):
    """Local development must not need a mail server."""
    monkeypatch.setattr(email.settings, "smtp_host", "")
    assert await email.send(email.Message(to="a@e.com", subject="s", body="b")) is False


async def test_the_message_is_logged_when_there_is_no_smtp_host(monkeypatch):
    """A developer still needs to be able to follow the invite link.

    The logger is captured directly rather than through ``caplog``: structlog
    is configured with its own processor chain and does not route through the
    stdlib handler pytest installs, so caplog.text would be empty and this
    would pass without proving anything.
    """
    monkeypatch.setattr(email.settings, "smtp_host", "")
    recorded: list[dict] = []
    monkeypatch.setattr(
        email.log, "info", lambda event, **kw: recorded.append({"event": event, **kw})
    )

    await email.send(email.Message(to="a@e.com", subject="s", body="the-link-here"))

    assert recorded[0]["event"] == "email.skipped_no_smtp_host"
    assert recorded[0]["body"] == "the-link-here"


async def test_a_dead_mail_server_does_not_raise(smtp_configured, monkeypatch):
    """The invite row is committed by the time this runs. A refused connection
    must not roll it back."""

    def _explode(*_args, **_kwargs):
        raise ConnectionRefusedError("no route to host")

    monkeypatch.setattr(email.smtplib, "SMTP", _explode)
    assert await email.send(email.Message(to="a@e.com", subject="s", body="b")) is False


async def test_a_successful_send_reports_true(smtp_configured, monkeypatch):
    sent = []

    class _FakeSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def starttls(self):
            pass

        def send_message(self, message):
            sent.append(message)

    monkeypatch.setattr(email.smtplib, "SMTP", _FakeSMTP)
    assert await email.send(email.Message(to="a@e.com", subject="Hi", body="body")) is True
    assert sent[0]["To"] == "a@e.com"
    assert sent[0]["Subject"] == "Hi"


# --- The invite link --------------------------------------------------------


def test_the_invite_link_comes_from_configuration(monkeypatch):
    """Never from a request Host header, which an attacker controls: that would
    let anyone trigger an invite email pointing at a domain they own."""
    monkeypatch.setattr(email.settings, "app_base_url", "https://hire.example.com")
    assert email.invite_link("tok123") == "https://hire.example.com/interview?token=tok123"


def test_a_trailing_slash_does_not_double_up(monkeypatch):
    monkeypatch.setattr(email.settings, "app_base_url", "https://hire.example.com/")
    assert "//interview" not in email.invite_link("t")


async def test_the_invite_email_carries_the_link_and_the_expiry(monkeypatch):
    monkeypatch.setattr(email.settings, "smtp_host", "")
    captured: list[email.Message] = []
    monkeypatch.setattr(email, "send", lambda m: _capture(captured, m))

    await email.send_invite(
        to="c@e.com",
        candidate_name="Ada",
        job_title="Staff Engineer",
        invite_token="tok",
        expires_hours=72,
    )
    body = captured[0].body
    assert "tok" in body
    assert "72 hours" in body
    assert "Ada" in body
    # The multi-use property is the one thing a candidate needs to be told.
    assert "rejoin" in body


async def test_an_unnamed_candidate_still_gets_a_greeting(monkeypatch):
    monkeypatch.setattr(email.settings, "smtp_host", "")
    captured: list[email.Message] = []
    monkeypatch.setattr(email, "send", lambda m: _capture(captured, m))

    await email.send_invite(
        to="c@e.com", candidate_name=None, job_title="Role", invite_token="t", expires_hours=1
    )
    assert "Hi there," in captured[0].body


async def _capture(sink: list, message) -> bool:
    sink.append(message)
    return True


# --- What notifications must not contain ------------------------------------


async def test_the_report_ready_email_carries_no_findings(monkeypatch):
    """This lands in an inbox, gets forwarded, and appears in lock-screen
    previews. The assessment of a named person belongs behind the login."""
    monkeypatch.setattr(email.settings, "smtp_host", "")
    captured: list[email.Message] = []
    monkeypatch.setattr(email, "send", lambda m: _capture(captured, m))

    await email.send_report_ready(
        to="r@e.com", candidate_name="Ada Lovelace", job_title="Staff Engineer"
    )
    text = f"{captured[0].subject} {captured[0].body}".lower()
    for word in ("score", "hire", "recommendation", "verdict", "flagged", "rating"):
        assert word not in text, f"the notification leaks {word!r}"


async def test_the_candidate_notification_carries_no_verdict(monkeypatch):
    monkeypatch.setattr(email.settings, "smtp_host", "")
    captured: list[email.Message] = []
    monkeypatch.setattr(email, "send", lambda m: _capture(captured, m))

    await email.send_candidate_feedback_ready(
        to="c@e.com", candidate_name="Ada", job_title="Staff Engineer"
    )
    text = f"{captured[0].subject} {captured[0].body}".lower()
    for word in ("score", "rating", "recommendation", "rejected", "unsuccessful"):
        assert word not in text, f"the notification leaks {word!r}"
