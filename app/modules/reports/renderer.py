"""HTML -> PDF -> S3, for both report types.

TWO TEMPLATES, ONE PER AUDIENCE, and no shared partials between them. A shared
"score block" include that the candidate template merely declines to call is
exactly the arrangement that leaks the first time someone refactors the layout.
The duplication is small and the guarantee is worth more than the lines saved.

``autoescape`` is on. Every string on these pages -- a candidate's name, a
transcript line, a model-written rationale -- is untrusted or model-authored,
and WeasyPrint parses the result as HTML. Without escaping, a candidate whose
name contains a tag reshapes the recruiter's report.

Rendering is CPU-bound and blocking, so it runs in a thread like everything
else that is. It is only ever called from a Celery worker, but a report
regenerated from the API path must not stall the event loop either.
"""

from __future__ import annotations

import asyncio
import uuid
from functools import lru_cache
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.core.config import settings
from app.integrations import storage

log = structlog.get_logger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "config" / "templates"

RECRUITER_TEMPLATE = "recruiter_report.html"
CANDIDATE_TEMPLATE = "candidate_report.html"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(default_for_string=True, default=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(template_name: str, view: object) -> str:
    return _environment().get_template(template_name).render(v=view)


def _to_pdf(html: str) -> bytes:
    """Blocking. Kept separate so tests can render HTML without paying for a PDF."""
    from weasyprint import HTML

    # base_url is required for WeasyPrint to resolve relative URLs. Pointing it
    # at the template directory rather than leaving it None means a stray
    # relative src cannot be resolved against the process's cwd, which in a
    # worker is the application root.
    return HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf()


async def render_pdf(template_name: str, view: object) -> bytes:
    html = render_html(template_name, view)
    return await asyncio.to_thread(_to_pdf, html)


def report_key(
    org_id: uuid.UUID, interview_id: uuid.UUID, audience: str, *, version: str | None = None
) -> str:
    """Server-chosen, org-prefixed, exactly as for resumes and frames.

    A fresh uuid per render rather than a stable key: re-rendering writes a new
    object, so a presigned URL a recruiter is already holding keeps resolving to
    the version they were reading instead of changing under them.
    """
    suffix = version or uuid.uuid4().hex
    return f"{org_id}/{interview_id}/{audience}-{suffix}.pdf"


async def publish(
    *, org_id: uuid.UUID, interview_id: uuid.UUID, audience: str, pdf: bytes
) -> str:
    key = report_key(org_id, interview_id, audience)
    await storage.put_bytes(
        bucket=settings.s3_bucket_reports,
        key=key,
        data=pdf,
        content_type="application/pdf",
    )
    log.info(
        "reports.published",
        interview_id=str(interview_id),
        audience=audience,
        bytes=len(pdf),
    )
    return key


async def download_url(key: str) -> str:
    """A short-lived link to one report.

    The key itself is never returned to a client. A bucket name is guessable and
    a key is structured, so handing both out is most of the way to an object.
    """
    return await storage.presign_get(
        bucket=settings.s3_bucket_reports,
        key=key,
        ttl_secs=settings.report_download_ttl_secs,
    )
