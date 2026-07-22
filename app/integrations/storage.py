"""S3-compatible storage (MinIO/R2): resumes, audio, frames, PDFs.

boto3 is synchronous. Every call here is wrapped in ``asyncio.to_thread`` rather
than pulling in aioboto3: the calls are infrequent (a presign, an upload at the
end of a session) and a second AWS SDK is a large dependency to carry for that.
Presigning in particular is pure local HMAC with no network at all.

WHY PRESIGNED URLs: a resume or a webcam frame goes browser -> S3 directly. The
alternative streams every upload through the API process, which turns a
CPU-light service into a bandwidth bottleneck and gives a candidate a way to tie
up request workers. The tradeoff is that the client learns a URL that can write
one object for a bounded window, which is why the key is server-chosen and the
TTL is 15 minutes.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import boto3
import structlog
from botocore.client import Config
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.exceptions import AppError

log = structlog.get_logger(__name__)

# Resumes only. Deliberately not "anything the parser might cope with": the
# parser in modules/resume handles exactly these two, and an allowlist that
# drifts ahead of it produces rows stuck in PARSING forever.
RESUME_CONTENT_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


class StorageError(AppError):
    status_code = 502
    code = "storage_unavailable"
    message = "Object storage is unavailable."


@dataclass(frozen=True, slots=True)
class PresignedUpload:
    url: str
    key: str
    bucket: str
    expires_in: int
    # Echoed back so the browser sends exactly the type the signature covers --
    # a mismatched Content-Type header makes S3 reject the PUT.
    content_type: str


@lru_cache(maxsize=1)
def _client() -> Any:
    """One client per process. boto3 clients are thread-safe."""
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url or None,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.s3_secret_access_key.get_secret_value(),
        # s3v4 is required for presigned PUTs to work against MinIO and R2.
        # "path" addressing keeps a bucket name in the path rather than in a
        # hostname, which a bare-IP MinIO endpoint cannot express.
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def reset_client() -> None:
    """Drop the cached client. For tests that repoint the endpoint."""
    _client.cache_clear()


def resume_key(org_id: uuid.UUID, candidate_id: uuid.UUID, extension: str) -> str:
    """Server-chosen, org-prefixed, and unguessable.

    The org prefix is what makes a future bucket policy or lifecycle rule
    expressible per tenant. The random component means knowing a candidate id is
    not enough to guess where their resume lives.
    """
    return f"{org_id}/{candidate_id}/{uuid.uuid4().hex}.{extension}"


async def presign_put(
    *, bucket: str, key: str, content_type: str, max_bytes: int | None = None
) -> PresignedUpload:
    """A URL the browser may PUT one object to.

    ``max_bytes`` is advisory on a plain presigned PUT -- only a POST policy can
    bind a size limit into the signature. The authoritative check is therefore
    the HEAD in ``head_object`` after the client reports completion, which is why
    the resume row is not marked uploaded until that check passes.
    """
    params: dict[str, Any] = {"Bucket": bucket, "Key": key, "ContentType": content_type}
    try:
        url = await asyncio.to_thread(
            _client().generate_presigned_url,
            "put_object",
            Params=params,
            ExpiresIn=settings.s3_presign_ttl_secs,
        )
    except ClientError as exc:
        raise StorageError() from exc

    log.info("presigned_put", bucket=bucket, key=key, max_bytes=max_bytes)
    return PresignedUpload(
        url=url,
        key=key,
        bucket=bucket,
        expires_in=settings.s3_presign_ttl_secs,
        content_type=content_type,
    )


async def presign_get(*, bucket: str, key: str, ttl_secs: int | None = None) -> str:
    """A time-limited read URL, for recruiter downloads and report links."""
    try:
        return await asyncio.to_thread(
            _client().generate_presigned_url,
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl_secs or settings.s3_presign_ttl_secs,
        )
    except ClientError as exc:
        raise StorageError() from exc


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    size: int
    content_type: str


async def head_object(*, bucket: str, key: str) -> ObjectInfo | None:
    """Size and type of an uploaded object, or None if it is not there.

    This is the server-side verification of a client-side upload: it is the only
    thing standing between a presigned PUT and a candidate claiming they uploaded
    a resume they never sent, or sending a 2GB file.
    """
    try:
        response = await asyncio.to_thread(_client().head_object, Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise StorageError() from exc
    return ObjectInfo(
        size=int(response["ContentLength"]),
        content_type=response.get("ContentType", "application/octet-stream"),
    )


async def put_bytes(
    *, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"
) -> str:
    """Server-side upload. For artifacts we generate: recordings, PDFs."""
    try:
        await asyncio.to_thread(
            _client().put_object, Bucket=bucket, Key=key, Body=data, ContentType=content_type
        )
    except ClientError as exc:
        raise StorageError() from exc
    log.info("object_stored", bucket=bucket, key=key, bytes=len(data))
    return key


async def get_bytes(*, bucket: str, key: str) -> bytes:
    """Whole-object read. Callers here handle documents and short audio, both of
    which fit in memory; anything larger should stream instead."""
    try:
        response = await asyncio.to_thread(_client().get_object, Bucket=bucket, Key=key)
        return await asyncio.to_thread(response["Body"].read)
    except ClientError as exc:
        raise StorageError() from exc


async def delete_object(*, bucket: str, key: str) -> None:
    """Idempotent: S3 does not error on a key that is already gone."""
    try:
        await asyncio.to_thread(_client().delete_object, Bucket=bucket, Key=key)
    except ClientError as exc:
        raise StorageError() from exc
