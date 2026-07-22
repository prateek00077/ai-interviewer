"""JWT sign/verify, password hashing, invite + ephemeral session tokens.

FOUR TOKEN TYPES, ONE SECRET, FOUR KEYS.

A ``typ`` claim checked with an ``if`` is one forgotten line away from letting a
72-hour invite token be presented as an access token. So each type is signed with
a *different* key, derived from JWT_SECRET by domain-separated HMAC. Presenting
the wrong type then fails at signature verification, before any claim is read --
the failure is structural rather than conditional.

Two further layers sit on top: a distinct ``aud`` per type (enforced by PyJWT)
and an explicit ``typ`` assertion after decode. Three independent barriers.
"""

import hashlib
import hmac
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final

import jwt
from pwdlib import PasswordHash

from app.core.config import settings
from app.core.exceptions import InvalidTokenError

# --- Password hashing -------------------------------------------------------

_password_hash = PasswordHash.recommended()  # Argon2id

# Verifying against a real hash on the unknown-user path keeps login's response
# time dominated by a KDF in both branches, so timing does not reveal whether an
# address is registered. Computed once at import.
_DUMMY_HASH: Final[str] = _password_hash.hash("dummy-password-for-timing-equalisation")


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return _password_hash.verify(password, hashed)
    except Exception:
        # A malformed stored hash must read as "wrong password", not a 500.
        return False


def verify_password_dummy() -> None:
    """Burn the same work as a real verify. Call on the unknown-user path."""
    _password_hash.verify("wrong", _DUMMY_HASH)


def verify_and_upgrade(password: str, hashed: str) -> tuple[bool, str | None]:
    """Returns ``(matched, upgraded_hash)``.

    ``upgraded_hash`` is None unless the stored hash used outdated parameters, in
    which case the login path should write the returned value back. Kept separate
    from ``verify_password`` so the boolean predicate stays a boolean -- a tuple
    is always truthy, and ``if verify(...)`` on one would let every password in.
    """
    try:
        return _password_hash.verify_and_update(password, hashed)
    except Exception:
        return False, None


# --- Token types and key derivation ----------------------------------------


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"
    INVITE = "invite"
    INTERVIEW = "interview"


class ActorKind(StrEnum):
    USER = "user"
    CANDIDATE = "candidate"


_KEY_CONTEXT: Final[bytes] = b"aii:jwt:v1"


def _key_for(token_type: TokenType) -> bytes:
    """One HMAC expand step is enough for domain separation between types."""
    return hmac.new(
        settings.jwt_secret.get_secret_value().encode(),
        _KEY_CONTEXT + b":" + token_type.value.encode(),
        hashlib.sha256,
    ).digest()


def _audience(token_type: TokenType) -> str:
    return f"aii:{token_type.value}"


_REQUIRED_CLAIMS: Final[list[str]] = ["exp", "iat", "sub", "jti", "aud", "iss", "typ"]


def _encode(token_type: TokenType, claims: dict[str, Any], ttl: timedelta) -> str:
    now = datetime.now(UTC)
    payload = {
        **claims,
        "typ": token_type.value,
        "aud": _audience(token_type),
        "iss": settings.jwt_issuer,
        "iat": now,
        "nbf": now,
        "exp": now + ttl,
    }
    payload.setdefault("jti", str(uuid.uuid4()))
    return jwt.encode(payload, _key_for(token_type), algorithm=settings.jwt_algorithm)


def decode_token(raw: str, expected: TokenType) -> dict[str, Any]:
    """Decode and fully validate a token of exactly one type."""
    try:
        claims: dict[str, Any] = jwt.decode(
            raw,
            _key_for(expected),
            # An allowlist, not the token's own header: this is what blocks
            # `alg: none` and HS/RS confusion.
            algorithms=[settings.jwt_algorithm],
            audience=_audience(expected),
            issuer=settings.jwt_issuer,
            # Without `require`, a token simply omitting `exp` validates forever.
            options={"require": _REQUIRED_CLAIMS},
            leeway=10,
        )
    except jwt.PyJWTError as exc:
        # The reason is logged by the caller, never returned to the client.
        raise InvalidTokenError() from exc

    if claims.get("typ") != expected.value:
        # Unreachable given the derived keys; kept so the invariant is explicit.
        raise InvalidTokenError()
    return claims


# --- Typed claim views ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AccessClaims:
    user_id: uuid.UUID
    org_id: uuid.UUID
    role: str
    jti: str

    @classmethod
    def parse(cls, c: dict[str, Any]) -> "AccessClaims":
        try:
            return cls(uuid.UUID(c["sub"]), uuid.UUID(c["org"]), c["role"], c["jti"])
        except (KeyError, ValueError) as exc:
            raise InvalidTokenError() from exc


@dataclass(frozen=True, slots=True)
class RefreshClaims:
    user_id: uuid.UUID
    org_id: uuid.UUID
    family_id: str
    jti: str

    @classmethod
    def parse(cls, c: dict[str, Any]) -> "RefreshClaims":
        try:
            return cls(uuid.UUID(c["sub"]), uuid.UUID(c["org"]), c["fam"], c["jti"])
        except (KeyError, ValueError) as exc:
            raise InvalidTokenError() from exc


@dataclass(frozen=True, slots=True)
class InviteClaims:
    invite_id: uuid.UUID
    org_id: uuid.UUID
    jti: str

    @classmethod
    def parse(cls, c: dict[str, Any]) -> "InviteClaims":
        try:
            return cls(uuid.UUID(c["sub"]), uuid.UUID(c["org"]), c["jti"])
        except (KeyError, ValueError) as exc:
            raise InvalidTokenError() from exc


@dataclass(frozen=True, slots=True)
class InterviewClaims:
    candidate_id: uuid.UUID
    org_id: uuid.UUID
    interview_id: uuid.UUID
    invite_id: uuid.UUID
    jti: str

    @classmethod
    def parse(cls, c: dict[str, Any]) -> "InterviewClaims":
        try:
            return cls(
                uuid.UUID(c["sub"]),
                uuid.UUID(c["org"]),
                uuid.UUID(c["iv"]),
                uuid.UUID(c["inv"]),
                c["jti"],
            )
        except (KeyError, ValueError) as exc:
            raise InvalidTokenError() from exc


# --- Minting ----------------------------------------------------------------


def create_access_token(
    *, user_id: uuid.UUID, org_id: uuid.UUID, role: str
) -> tuple[str, int]:
    """Returns (token, expires_in_seconds)."""
    ttl = timedelta(minutes=settings.access_token_ttl_minutes)
    token = _encode(
        TokenType.ACCESS,
        {
            "sub": str(user_id),
            "org": str(org_id),
            "role": role,
            "act": ActorKind.USER.value,
        },
        ttl,
    )
    return token, int(ttl.total_seconds())


def create_refresh_token(
    *, user_id: uuid.UUID, org_id: uuid.UUID, family_id: str, jti: str | None = None
) -> tuple[str, str]:
    """Returns (token, jti).

    Deliberately carries no ``role``: the role is re-read from the database on
    every rotation, so a demotion takes effect within one access-token lifetime
    instead of persisting for the full 14 days.
    """
    token_jti = jti or str(uuid.uuid4())
    token = _encode(
        TokenType.REFRESH,
        {
            "sub": str(user_id),
            "org": str(org_id),
            "fam": family_id,
            "jti": token_jti,
        },
        timedelta(days=settings.refresh_token_ttl_days),
    )
    return token, token_jti


def create_invite_token(
    *, invite_id: uuid.UUID, org_id: uuid.UUID, jti: str
) -> str:
    """A pointer to an invite row, not a payload.

    ``jti`` must match ``Invite.jti``, so revoking the row invalidates the token
    immediately rather than at expiry.
    """
    return _encode(
        TokenType.INVITE,
        {"sub": str(invite_id), "org": str(org_id), "jti": jti},
        timedelta(hours=settings.invite_ttl_hours),
    )


def create_interview_token(
    *,
    candidate_id: uuid.UUID,
    org_id: uuid.UUID,
    interview_id: uuid.UUID,
    invite_id: uuid.UUID,
) -> tuple[str, int]:
    """The ephemeral token the browser trades for a voice connection."""
    ttl = timedelta(minutes=settings.interview_token_ttl_minutes)
    token = _encode(
        TokenType.INTERVIEW,
        {
            "sub": str(candidate_id),
            "org": str(org_id),
            "iv": str(interview_id),
            "inv": str(invite_id),
            "role": "CANDIDATE",
            "act": ActorKind.CANDIDATE.value,
        },
        ttl,
    )
    return token, int(ttl.total_seconds())
