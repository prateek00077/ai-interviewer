"""POST login, refresh, register-org.

Routes are thin on purpose: they validate, delegate, and shape a response. Every
decision that matters -- uniform failures, rotation, redemption atomicity --
lives in ``app.modules.auth`` where it can be tested without HTTP.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    Principal,
    ScopedSession,
    get_current_user,
    get_token_store,
    get_unscoped_db,
    login_rate_limit,
    register_rate_limit,
    require_role,
)
from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.integrations import email
from app.models.user import UserRole
from app.modules.auth import invites as invite_service
from app.modules.auth import service as auth_service
from app.modules.auth.tokens import RefreshTokenStore
from app.modules.jobs import service as jobs_service
from app.schemas.auth import (
    CreateInviteRequest,
    InterviewTokenResponse,
    InviteResponse,
    LoginRequest,
    LogoutRequest,
    PrincipalResponse,
    RedeemInviteRequest,
    RefreshRequest,
    RegisterOrgRequest,
    RegisterOrgResponse,
    TokenResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])

Store = Annotated[RefreshTokenStore, Depends(get_token_store)]
UnscopedDb = Annotated[AsyncSession, Depends(get_unscoped_db)]
CurrentUser = Annotated[Principal, Depends(get_current_user)]


def _tokens(pair: auth_service.TokenPair) -> TokenResponse:
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        token_type=pair.token_type,
        expires_in=pair.expires_in,
    )


@router.post(
    "/register-org",
    response_model=RegisterOrgResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(register_rate_limit)],
    summary="Create a tenant and its first admin",
)
async def register_org(payload: RegisterOrgRequest, store: Store) -> RegisterOrgResponse:
    principal = await auth_service.register_org(
        org_name=payload.org_name,
        slug=payload.slug,
        admin_email=payload.admin_email,
        admin_password=payload.admin_password,
        admin_full_name=payload.admin_full_name,
    )
    pair = await auth_service.issue_token_pair(store, principal)
    return RegisterOrgResponse(
        org_id=principal.org_id, user_id=principal.user_id, tokens=_tokens(pair)
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(login_rate_limit)],
    summary="Exchange credentials for a token pair",
)
async def login(payload: LoginRequest, db: UnscopedDb, store: Store) -> TokenResponse:
    # Unknown email, wrong password, inactive user and inactive org all surface
    # here as the same 401 with the same body.
    principal = await auth_service.authenticate(
        db, email=payload.email, password=payload.password
    )
    return _tokens(await auth_service.issue_token_pair(store, principal))


@router.post("/refresh", response_model=TokenResponse, summary="Rotate a refresh token")
async def refresh(payload: RefreshRequest, store: Store) -> TokenResponse:
    return _tokens(await auth_service.rotate_refresh(store, payload.refresh_token))


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke one session",
)
async def logout(payload: LogoutRequest, store: Store) -> None:
    # Idempotent, and deliberately 204 even for an already-dead token: a 401 here
    # would tell an attacker which tokens are still live.
    await auth_service.logout(store, payload.refresh_token)


@router.post(
    "/logout-all",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke every session for the caller",
)
async def logout_all(principal: CurrentUser, store: Store) -> None:
    await auth_service.logout_all(store, principal.actor_id)


@router.get("/me", response_model=PrincipalResponse, summary="Who the bearer token is")
async def me(principal: CurrentUser) -> PrincipalResponse:
    return PrincipalResponse(
        org_id=principal.org_id,
        actor_kind=principal.actor_kind.value,
        actor_id=principal.actor_id,
        role=principal.role,
    )


@router.post(
    "/invites",
    response_model=InviteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Invite a candidate to an interview",
)
async def create_invite(
    payload: CreateInviteRequest,
    db: ScopedSession,
    principal: Annotated[Principal, Depends(require_role(UserRole.ADMIN, UserRole.RECRUITER))],
) -> InviteResponse:
    created = await invite_service.create_invite(
        db,
        org_id=principal.org_id,
        created_by_user_id=principal.actor_id,
        candidate_email=payload.candidate_email,
        candidate_name=payload.candidate_name,
        job_id=payload.job_id,
        max_redemptions=payload.max_redemptions,
    )

    # After the invite is created, and never allowed to undo it. The row is the
    # work; the email is the announcement. A mail server that is down must not
    # roll back an invite the recruiter has just been told about -- and the
    # token comes back in the response either way, so the link is recoverable
    # by hand.
    job_title = "an interview"
    if payload.job_id is not None:
        try:
            job_title = (await jobs_service.get_job(db, payload.job_id)).title
        except NotFoundError:
            pass
    await email.send_invite(
        to=payload.candidate_email,
        candidate_name=payload.candidate_name,
        job_title=job_title,
        invite_token=created.token,
        expires_hours=settings.invite_ttl_hours,
    )

    return InviteResponse(
        invite_id=created.invite_id,
        interview_id=created.interview_id,
        candidate_id=created.candidate_id,
        invite_token=created.token,
        expires_at=created.expires_at,
    )


@router.post(
    "/invites/redeem",
    response_model=InterviewTokenResponse,
    summary="Trade an invite link for a short-lived interview token",
)
async def redeem_invite(payload: RedeemInviteRequest) -> InterviewTokenResponse:
    # Public: the candidate has no session yet. The org comes from the token's
    # own signed claim, never from the request.
    redeemed = await invite_service.redeem_invite(payload.invite_token)
    return InterviewTokenResponse(
        interview_token=redeemed.interview_token,
        expires_in=redeemed.expires_in,
        interview_id=redeemed.interview_id,
        candidate_id=redeemed.candidate_id,
    )
