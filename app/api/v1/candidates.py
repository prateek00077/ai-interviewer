"""Candidate CRUD and resume upload.

Two audiences share this router, and the split is strict:

- ``/candidates/...`` is recruiter-facing. Both ADMIN and RECRUITER may read and
  write, because sourcing is the recruiter's day job.
- ``/candidates/me/...`` is candidate-facing, reached only with an interview
  token, and always acts on the candidate the token names. There is no path
  parameter to tamper with.

Resume upload is candidate-only by design: the candidate is the one holding the
file. It is a two-step handshake -- presign, then complete -- because the bytes
go browser-to-S3 directly and the server has to verify afterwards what actually
landed there.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import (
    Principal,
    ScopedSession,
    get_current_candidate,
    require_role,
)
from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.models.user import UserRole
from app.modules.resume import service as resume_service
from app.modules.users import service as users_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, Page
from app.schemas.resume import (
    CandidateResumeRead,
    ResumeDownload,
    ResumePresignRequest,
    ResumePresignResponse,
    ResumeRead,
)
from app.schemas.user import CandidateCreate, CandidateRead, CandidateUpdate
from app.workers.tasks.resume_tasks import process_resume

router = APIRouter(prefix="/candidates", tags=["candidates"])

Recruiter = Annotated[Principal, Depends(require_role(UserRole.ADMIN, UserRole.RECRUITER))]
CurrentCandidate = Annotated[Principal, Depends(get_current_candidate)]


@router.get("", response_model=Page[CandidateRead], summary="List candidates")
async def list_candidates(
    db: ScopedSession,
    _: Recruiter,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
    search: Annotated[str | None, Query(max_length=200)] = None,
) -> Page[CandidateRead]:
    items, total = await users_service.list_candidates(
        db, limit=limit, offset=offset, search=search
    )
    return Page[CandidateRead](
        items=[CandidateRead.model_validate(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "",
    response_model=CandidateRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a candidate",
)
async def create_candidate(
    payload: CandidateCreate, db: ScopedSession, principal: Recruiter
) -> CandidateRead:
    # org_id comes from the token, never from the body.
    candidate = await users_service.create_candidate(
        db,
        org_id=principal.org_id,
        email=payload.email,
        full_name=payload.full_name,
        phone=payload.phone,
        external_ref=payload.external_ref,
    )
    return CandidateRead.model_validate(candidate)


# --- Candidate-facing resume upload -----------------------------------------
#
# These are declared BEFORE /{candidate_id}: FastAPI matches in declaration
# order, so a later "/me" would be swallowed by the UUID path parameter.


@router.post(
    "/me/resume/presign",
    response_model=ResumePresignResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Get a URL to upload a resume to",
)
async def presign_own_resume(
    payload: ResumePresignRequest, db: ScopedSession, principal: CurrentCandidate
) -> ResumePresignResponse:
    # The candidate is whoever the interview token says. No path or body field
    # can point this at someone else.
    resume, upload = await resume_service.start_upload(
        db,
        org_id=principal.org_id,
        candidate_id=principal.actor_id,
        filename=payload.filename,
        content_type=payload.content_type,
        declared_size=payload.declared_size,
    )
    return ResumePresignResponse(
        resume_id=resume.id,
        upload_url=upload.url,
        content_type=upload.content_type,
        expires_in=upload.expires_in,
        max_bytes=settings.max_resume_bytes,
    )


@router.post(
    "/me/resume/{resume_id}/complete",
    response_model=CandidateResumeRead,
    summary="Confirm the upload finished",
)
async def complete_own_resume(
    resume_id: uuid.UUID, db: ScopedSession, principal: CurrentCandidate
) -> CandidateResumeRead:
    resume, transitioned = await resume_service.complete_upload(
        db, resume_id=resume_id, candidate_id=principal.actor_id
    )
    # Keyed on the transition, not on the resulting status: a retried /complete
    # also sees UPLOADED, and enqueueing off that would run the pipeline twice.
    if transitioned:
        # Commit before enqueueing. The request's session would otherwise commit
        # at teardown, and the worker could pick the job up first and read a row
        # that is not visible to it yet.
        await db.commit()
        process_resume.delay(str(principal.org_id), str(resume.id))
    return CandidateResumeRead.model_validate(resume)


@router.get(
    "/me/resume",
    response_model=list[CandidateResumeRead],
    summary="The candidate's own uploads",
)
async def list_own_resumes(
    db: ScopedSession, principal: CurrentCandidate
) -> list[CandidateResumeRead]:
    rows = await resume_service.list_for_candidate(db, principal.actor_id)
    return [CandidateResumeRead.model_validate(r) for r in rows]


# --- Recruiter-facing resume access -----------------------------------------


@router.get(
    "/{candidate_id}/resumes",
    response_model=list[ResumeRead],
    summary="A candidate's resumes, with parsed fields",
)
async def list_candidate_resumes(
    candidate_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> list[ResumeRead]:
    await users_service.get_candidate(db, candidate_id)  # 404 for another org
    rows = await resume_service.list_for_candidate(db, candidate_id)
    return [ResumeRead.model_validate(r) for r in rows]


@router.get(
    "/{candidate_id}/resumes/{resume_id}/download",
    response_model=ResumeDownload,
    summary="A time-limited link to the original file",
)
async def download_candidate_resume(
    candidate_id: uuid.UUID, resume_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> ResumeDownload:
    resume = await resume_service.get_resume(db, resume_id)
    if resume.candidate_id != candidate_id:
        raise NotFoundError("Resume not found.", resume_id=str(resume_id))
    url, expires_in = await resume_service.download_url(db, resume_id)
    return ResumeDownload(url=url, expires_in=expires_in)


@router.get("/{candidate_id}", response_model=CandidateRead, summary="Get one candidate")
async def get_candidate(
    candidate_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> CandidateRead:
    return CandidateRead.model_validate(await users_service.get_candidate(db, candidate_id))


@router.patch("/{candidate_id}", response_model=CandidateRead, summary="Update a candidate")
async def update_candidate(
    candidate_id: uuid.UUID, payload: CandidateUpdate, db: ScopedSession, _: Recruiter
) -> CandidateRead:
    candidate = await users_service.update_candidate(
        db,
        candidate_id=candidate_id,
        full_name=payload.full_name,
        phone=payload.phone,
        external_ref=payload.external_ref,
        fields_set=payload.model_fields_set,
    )
    return CandidateRead.model_validate(candidate)


@router.delete(
    "/{candidate_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a candidate and their interviews",
)
async def delete_candidate(candidate_id: uuid.UUID, db: ScopedSession, _: Recruiter) -> None:
    # A real delete, cascading: this is the erasure path for personal data.
    await users_service.delete_candidate(db, candidate_id)
