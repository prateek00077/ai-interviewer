"""Magic invite link -> short-lived interview token.

Redemption is the interesting half. It is expressed as **one conditional UPDATE**
rather than a read, a decision, and a write: every eligibility rule lives in the
WHERE clause, so simultaneous redemptions serialize on the row lock and the
fourth attempt at a three-use invite cannot slip through between another
request's SELECT and its UPDATE.

Zero rows updated means unusable, and the caller is never told which rule
failed -- expired, revoked, exhausted and forged all return the same 410. The
distinction is only useful to someone probing links they were not given.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import InviteUnusableError, NotFoundError
from app.core.security import (
    InviteClaims,
    TokenType,
    create_interview_token,
    create_invite_token,
    decode_token,
)
from app.db.session import tenant_session
from app.models.interview import Interview, InterviewStatus, Invite, InviteStatus
from app.models.user import Candidate

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CreatedInvite:
    invite_id: uuid.UUID
    interview_id: uuid.UUID
    candidate_id: uuid.UUID
    token: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class RedeemedInvite:
    interview_token: str
    expires_in: int
    interview_id: uuid.UUID
    candidate_id: uuid.UUID


# Every eligibility rule is a predicate here, not a branch in Python.
_REDEEM_SQL = text(
    """
    UPDATE invites
       SET redemption_count = redemption_count + 1,
           status           = 'REDEEMED',
           redeemed_at      = coalesce(redeemed_at, now())
     WHERE id = :invite_id
       AND jti = :jti
       AND status IN ('PENDING', 'REDEEMED')
       AND revoked_at IS NULL
       AND expires_at > now()
       AND redemption_count < max_redemptions
 RETURNING interview_id, candidate_id, org_id
    """
)


async def create_invite(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    created_by_user_id: uuid.UUID,
    candidate_email: str,
    candidate_name: str | None = None,
    job_id: uuid.UUID | None = None,
    max_redemptions: int | None = None,
) -> CreatedInvite:
    """Upsert the candidate, open an interview, and sign a pointer token.

    ``session`` must already be org-scoped: RLS supplies the tenant boundary, so
    nothing here filters on ``org_id`` by hand.
    """
    candidate = (
        await session.execute(select(Candidate).where(Candidate.email == candidate_email))
    ).scalar_one_or_none()

    if candidate is None:
        candidate = Candidate(
            id=uuid.uuid4(), org_id=org_id, email=candidate_email, full_name=candidate_name
        )
        session.add(candidate)
        await session.flush()
    elif candidate_name and not candidate.full_name:
        candidate.full_name = candidate_name

    interview = Interview(
        id=uuid.uuid4(),
        org_id=org_id,
        candidate_id=candidate.id,
        job_id=job_id,
        status=InterviewStatus.INVITED,
    )
    session.add(interview)
    await session.flush()

    invite_id = uuid.uuid4()
    jti = uuid.uuid4()
    expires_at = datetime.now(UTC) + timedelta(hours=settings.invite_ttl_hours)
    session.add(
        Invite(
            id=invite_id,
            org_id=org_id,
            interview_id=interview.id,
            candidate_id=candidate.id,
            jti=jti,
            status=InviteStatus.PENDING,
            expires_at=expires_at,
            max_redemptions=max_redemptions or settings.invite_max_redemptions,
            created_by_user_id=created_by_user_id,
        )
    )
    await session.flush()

    # The jti in the token must equal Invite.jti, which is what makes revoking
    # the row invalidate the token instantly rather than at expiry.
    token = create_invite_token(invite_id=invite_id, org_id=org_id, jti=str(jti))

    log.info(
        "auth.invite_created",
        invite_id=str(invite_id),
        interview_id=str(interview.id),
        org_id=str(org_id),
    )
    return CreatedInvite(
        invite_id=invite_id,
        interview_id=interview.id,
        candidate_id=candidate.id,
        token=token,
        expires_at=expires_at,
    )


async def redeem_invite(raw_token: str) -> RedeemedInvite:
    """Trade an invite token for a short-lived interview token.

    Public: the caller has no session yet. The org comes from the token's own
    signed ``org`` claim, so it cannot be chosen by the requester.
    """
    try:
        claims = InviteClaims.parse(decode_token(raw_token, TokenType.INVITE))
    except Exception as exc:
        # A malformed or wrong-type invite link is "no longer valid" too --
        # 401 vs 410 here would separate forged links from expired ones.
        raise InviteUnusableError() from exc

    # actor_kind must be 'user': invites is a USER_ONLY table, and the caller is
    # not yet a candidate as far as the database is concerned.
    async with tenant_session(claims.org_id, "user", None) as session:
        row = (
            await session.execute(
                _REDEEM_SQL, {"invite_id": claims.invite_id, "jti": claims.jti}
            )
        ).one_or_none()

        if row is None:
            log.info(
                "auth.invite_rejected",
                invite_id=str(claims.invite_id),
                org_id=str(claims.org_id),
            )
            raise InviteUnusableError()

        interview = await session.get(Interview, row.interview_id)
        if interview is not None and interview.status is InterviewStatus.INVITED:
            interview.status = InterviewStatus.IN_PROGRESS

    token, expires_in = create_interview_token(
        candidate_id=row.candidate_id,
        org_id=row.org_id,
        interview_id=row.interview_id,
        invite_id=claims.invite_id,
    )
    log.info("auth.invite_redeemed", invite_id=str(claims.invite_id), org_id=str(row.org_id))
    return RedeemedInvite(
        interview_token=token,
        expires_in=expires_in,
        interview_id=row.interview_id,
        candidate_id=row.candidate_id,
    )


async def revoke_invite(session: AsyncSession, invite_id: uuid.UUID) -> None:
    """Kill a link before it expires. Takes effect on the next redemption."""
    invite = await session.get(Invite, invite_id)
    if invite is None:
        # RLS already scoped the lookup, so "not in this org" reads as not found.
        raise NotFoundError("Invite not found.")
    invite.status = InviteStatus.REVOKED
    invite.revoked_at = datetime.now(UTC)
    log.info("auth.invite_revoked", invite_id=str(invite_id))
