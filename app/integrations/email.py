"""Outbound email: the invite link and the report-ready notice.

WHY SENDING NEVER RAISES. Every caller here is doing something else that has
already succeeded -- an invite row is committed, a report is rendered and in S3
-- and email is the notification, not the work. A mail server that is down or
slow must not roll back an invite the recruiter just created, and it must not
fail a Celery task into a retry loop that re-renders a PDF each time. Failures
are logged with enough context to resend by hand.

WHY ``smtplib`` IN A THREAD rather than aiosmtplib: the same reasoning as boto3
in ``storage``. Two or three messages per interview do not justify a second
protocol stack, and the blocking client in a thread is a smaller thing to be
wrong about.

WHY AN EMPTY ``SMTP_HOST`` IS A NO-OP rather than an error: local development
must not need a mail server. The message is logged in full instead, so a
developer can still see the invite link and follow it.
"""

from __future__ import annotations

import asyncio
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

import structlog

from app.core.config import settings

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Message:
    to: str
    subject: str
    body: str


def _build(message: Message) -> EmailMessage:
    email = EmailMessage()
    email["From"] = settings.email_from
    email["To"] = message.to
    email["Subject"] = message.subject
    email.set_content(message.body)
    return email


def _send_blocking(message: Message) -> None:
    email = _build(message)
    with smtplib.SMTP(
        settings.smtp_host, settings.smtp_port, timeout=settings.smtp_timeout_secs
    ) as client:
        if settings.smtp_starttls:
            client.starttls()
        if settings.smtp_user:
            client.login(settings.smtp_user, settings.smtp_password.get_secret_value())
        client.send_message(email)


async def send(message: Message) -> bool:
    """Deliver one message. Returns whether it was actually sent."""
    if not settings.smtp_host:
        log.info(
            "email.skipped_no_smtp_host",
            to=message.to,
            subject=message.subject,
            body=message.body,
        )
        return False

    try:
        await asyncio.to_thread(_send_blocking, message)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        log.error(
            "email.send_failed",
            to=message.to,
            subject=message.subject,
            error=str(exc)[:300],
        )
        return False

    log.info("email.sent", to=message.to, subject=message.subject)
    return True


# --- The two messages this product sends ------------------------------------


def invite_link(invite_token: str) -> str:
    """The URL a candidate follows.

    Built from configuration, never from a request header: a Host header is
    attacker-controlled, and using one here would let anyone trigger an invite
    email pointing at a domain they own.
    """
    return f"{settings.app_base_url.rstrip('/')}/interview?token={invite_token}"


async def send_invite(
    *, to: str, candidate_name: str | None, job_title: str, invite_token: str, expires_hours: int
) -> bool:
    name = (candidate_name or "there").strip() or "there"
    link = invite_link(invite_token)
    return await send(
        Message(
            to=to,
            subject=f"Your interview for {job_title}",
            body=(
                f"Hi {name},\n\n"
                f"You have been invited to an interview for {job_title}.\n\n"
                f"Join here:\n{link}\n\n"
                f"The link is valid for {expires_hours} hours. You can use it more "
                "than once, so if your connection drops you can rejoin with the "
                "same link.\n\n"
                "You will need a working microphone, and a camera if the role "
                "requires proctoring.\n"
            ),
        )
    )


async def send_report_ready(*, to: str, candidate_name: str, job_title: str) -> bool:
    """Tells a recruiter the report exists. Carries no findings.

    Deliberately contains no score, band or recommendation: this lands in an
    inbox, gets forwarded, and shows up in notification previews on lock
    screens. The assessment of a named person belongs behind the login.
    """
    return await send(
        Message(
            to=to,
            subject=f"Interview report ready: {candidate_name} ({job_title})",
            body=(
                f"The interview report for {candidate_name} ({job_title}) is ready.\n\n"
                "Sign in to review it.\n"
            ),
        )
    )


async def send_candidate_feedback_ready(*, to: str, candidate_name: str, job_title: str) -> bool:
    name = (candidate_name or "there").strip() or "there"
    return await send(
        Message(
            to=to,
            subject=f"Your interview feedback for {job_title}",
            body=(
                f"Hi {name},\n\n"
                f"Thank you for interviewing for {job_title}. Your feedback summary "
                "is ready and is available from the link you used to join.\n\n"
                "It covers what came across well and what would be worth developing. "
                "The hiring team will be in touch separately about next steps.\n"
            ),
        )
    )
