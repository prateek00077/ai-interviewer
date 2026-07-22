"""Proves a token of one type can never be verified as another.

The design claim in ``core.security`` is that type confusion is *structural*:
each type is signed with a key derived from JWT_SECRET by domain-separated HMAC,
so a wrong-type token dies at signature verification rather than at an ``if``.
The matrix below is what keeps that claim honest -- if someone ever "simplifies"
``_key_for`` to return the raw secret, these twelve cases go red.
"""

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import settings
from app.core.exceptions import InvalidTokenError
from app.core.security import (
    AccessClaims,
    InterviewClaims,
    InviteClaims,
    RefreshClaims,
    TokenType,
    _audience,
    _key_for,
    create_access_token,
    create_interview_token,
    create_invite_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_and_upgrade,
    verify_password,
    verify_password_dummy,
)

ORG = uuid.uuid4()
USER = uuid.uuid4()
CANDIDATE = uuid.uuid4()
INTERVIEW = uuid.uuid4()
INVITE = uuid.uuid4()


def _mint(token_type: TokenType) -> str:
    """One valid token of each type, built through the public minting API."""
    if token_type is TokenType.ACCESS:
        return create_access_token(user_id=USER, org_id=ORG, role="ADMIN")[0]
    if token_type is TokenType.REFRESH:
        return create_refresh_token(user_id=USER, org_id=ORG, family_id="fam-1")[0]
    if token_type is TokenType.INVITE:
        return create_invite_token(invite_id=INVITE, org_id=ORG, jti=str(uuid.uuid4()))
    return create_interview_token(
        candidate_id=CANDIDATE, org_id=ORG, interview_id=INTERVIEW, invite_id=INVITE
    )[0]


# --- The matrix -------------------------------------------------------------


@pytest.mark.parametrize("minted", list(TokenType))
def test_token_verifies_as_its_own_type(minted: TokenType):
    claims = decode_token(_mint(minted), minted)
    assert claims["typ"] == minted.value
    assert claims["aud"] == _audience(minted)
    assert claims["iss"] == settings.jwt_issuer


@pytest.mark.parametrize(
    ("minted", "presented"),
    [(m, p) for m in TokenType for p in TokenType if m is not p],
)
def test_cross_type_tokens_are_rejected(minted: TokenType, presented: TokenType):
    """All twelve off-diagonal cells. This is the whole point of the design."""
    with pytest.raises(InvalidTokenError):
        decode_token(_mint(minted), presented)


@pytest.mark.parametrize(
    ("a", "b"),
    [(a, b) for a in TokenType for b in TokenType if a.value < b.value],
)
def test_derived_keys_are_pairwise_distinct(a: TokenType, b: TokenType):
    assert _key_for(a) != _key_for(b)
    assert _key_for(a) != settings.jwt_secret.get_secret_value().encode()


# --- Forgery and tampering --------------------------------------------------


def test_alg_none_is_rejected():
    """An unsigned token forged with the right claims must not validate."""
    now = datetime.now(UTC)
    forged = jwt.encode(
        {
            "sub": str(USER),
            "org": str(ORG),
            "role": "ADMIN",
            "jti": str(uuid.uuid4()),
            "typ": TokenType.ACCESS.value,
            "aud": _audience(TokenType.ACCESS),
            "iss": settings.jwt_issuer,
            "iat": now,
            "exp": now + timedelta(minutes=30),
        },
        key="",
        algorithm="none",
    )
    with pytest.raises(InvalidTokenError):
        decode_token(forged, TokenType.ACCESS)


def test_token_signed_with_the_raw_secret_is_rejected():
    """Guards the derivation itself: the bare JWT_SECRET must not be a valid key."""
    now = datetime.now(UTC)
    forged = jwt.encode(
        {
            "sub": str(USER),
            "org": str(ORG),
            "role": "ADMIN",
            "jti": str(uuid.uuid4()),
            "typ": TokenType.ACCESS.value,
            "aud": _audience(TokenType.ACCESS),
            "iss": settings.jwt_issuer,
            "iat": now,
            "exp": now + timedelta(minutes=30),
        },
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(InvalidTokenError):
        decode_token(forged, TokenType.ACCESS)


def test_tampered_payload_is_rejected():
    """Flip a byte in the payload segment; the signature must stop matching."""
    header, payload, signature = _mint(TokenType.ACCESS).split(".")
    mangled = payload[:-1] + ("A" if payload[-1] != "A" else "B")
    with pytest.raises(InvalidTokenError):
        decode_token(f"{header}.{mangled}.{signature}", TokenType.ACCESS)


@pytest.mark.parametrize("garbage", ["", "not-a-jwt", "a.b.c", "a.b"])
def test_malformed_tokens_raise_invalid_token(garbage: str):
    with pytest.raises(InvalidTokenError):
        decode_token(garbage, TokenType.ACCESS)


def test_expired_token_is_rejected():
    now = datetime.now(UTC) - timedelta(hours=2)
    stale = jwt.encode(
        {
            "sub": str(USER),
            "org": str(ORG),
            "role": "ADMIN",
            "jti": str(uuid.uuid4()),
            "typ": TokenType.ACCESS.value,
            "aud": _audience(TokenType.ACCESS),
            "iss": settings.jwt_issuer,
            "iat": now,
            "exp": now + timedelta(minutes=1),
        },
        _key_for(TokenType.ACCESS),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(InvalidTokenError):
        decode_token(stale, TokenType.ACCESS)


def test_token_without_exp_does_not_validate_forever():
    """``options={"require": [...]}`` is what makes this fail."""
    forever = jwt.encode(
        {
            "sub": str(USER),
            "org": str(ORG),
            "role": "ADMIN",
            "jti": str(uuid.uuid4()),
            "typ": TokenType.ACCESS.value,
            "aud": _audience(TokenType.ACCESS),
            "iss": settings.jwt_issuer,
            "iat": datetime.now(UTC),
        },
        _key_for(TokenType.ACCESS),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(InvalidTokenError):
        decode_token(forever, TokenType.ACCESS)


def test_foreign_issuer_is_rejected():
    now = datetime.now(UTC)
    foreign = jwt.encode(
        {
            "sub": str(USER),
            "org": str(ORG),
            "role": "ADMIN",
            "jti": str(uuid.uuid4()),
            "typ": TokenType.ACCESS.value,
            "aud": _audience(TokenType.ACCESS),
            "iss": "some-other-service",
            "iat": now,
            "exp": now + timedelta(minutes=30),
        },
        _key_for(TokenType.ACCESS),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(InvalidTokenError):
        decode_token(foreign, TokenType.ACCESS)


# --- Claim views ------------------------------------------------------------


def test_access_claims_round_trip():
    claims = AccessClaims.parse(decode_token(_mint(TokenType.ACCESS), TokenType.ACCESS))
    assert (claims.user_id, claims.org_id, claims.role) == (USER, ORG, "ADMIN")


def test_refresh_token_carries_no_role():
    """A demotion must not survive in a 14-day token."""
    raw = decode_token(_mint(TokenType.REFRESH), TokenType.REFRESH)
    assert "role" not in raw
    claims = RefreshClaims.parse(raw)
    assert (claims.user_id, claims.org_id, claims.family_id) == (USER, ORG, "fam-1")


def test_refresh_rotation_keeps_family_and_changes_jti():
    _, jti_one = create_refresh_token(user_id=USER, org_id=ORG, family_id="fam-1")
    token_two, jti_two = create_refresh_token(user_id=USER, org_id=ORG, family_id="fam-1")
    assert jti_one != jti_two
    assert RefreshClaims.parse(decode_token(token_two, TokenType.REFRESH)).family_id == "fam-1"


def test_invite_claims_point_at_a_row():
    jti = str(uuid.uuid4())
    token = create_invite_token(invite_id=INVITE, org_id=ORG, jti=jti)
    claims = InviteClaims.parse(decode_token(token, TokenType.INVITE))
    # jti must match Invite.jti so revoking the row kills the token instantly.
    assert (claims.invite_id, claims.org_id, claims.jti) == (INVITE, ORG, jti)


def test_interview_claims_round_trip():
    token, expires_in = create_interview_token(
        candidate_id=CANDIDATE, org_id=ORG, interview_id=INTERVIEW, invite_id=INVITE
    )
    assert expires_in == settings.interview_token_ttl_minutes * 60
    decoded = decode_token(token, TokenType.INTERVIEW)
    assert decoded["act"] == "candidate"
    claims = InterviewClaims.parse(decoded)
    assert (claims.candidate_id, claims.interview_id, claims.invite_id) == (
        CANDIDATE,
        INTERVIEW,
        INVITE,
    )


@pytest.mark.parametrize(
    ("view", "token_type"),
    [
        (AccessClaims, TokenType.ACCESS),
        (RefreshClaims, TokenType.REFRESH),
        (InviteClaims, TokenType.INVITE),
        (InterviewClaims, TokenType.INTERVIEW),
    ],
)
def test_claim_views_reject_missing_fields(view, token_type):
    with pytest.raises(InvalidTokenError):
        view.parse({"jti": "x"})


def test_claim_views_reject_non_uuid_subjects():
    with pytest.raises(InvalidTokenError):
        AccessClaims.parse({"sub": "not-a-uuid", "org": str(ORG), "role": "ADMIN", "jti": "x"})


# --- Passwords --------------------------------------------------------------


def test_password_hash_verifies_and_is_salted():
    hashed = hash_password("correct-horse-battery")
    assert verify_password("correct-horse-battery", hashed)
    assert not verify_password("Correct-horse-battery", hashed)
    # Argon2id, and the plaintext must not appear anywhere in the digest.
    assert hashed.startswith("$argon2id$")
    assert "correct-horse-battery" not in hashed
    # Distinct salts, so two identical passwords do not collide in the DB.
    assert hashed != hash_password("correct-horse-battery")


@pytest.mark.parametrize("stored", ["", "not-a-hash", "$argon2id$broken"])
def test_malformed_stored_hash_reads_as_wrong_password(stored: str):
    """A corrupt row must be a 401, never a 500 that leaks a stack trace."""
    assert verify_password("anything", stored) is False


def test_dummy_verify_is_available_for_the_unknown_user_path():
    # Must not raise: login calls this to keep both branches KDF-bound.
    verify_password_dummy()


def test_verify_and_upgrade_leaves_current_hashes_alone():
    matched, upgraded = verify_and_upgrade("whatever", hash_password("whatever"))
    assert matched is True
    assert upgraded is None, "a freshly minted hash must not ask to be rewritten"


def test_verify_and_upgrade_reports_no_match_without_offering_a_hash():
    matched, upgraded = verify_and_upgrade("wrong", hash_password("whatever"))
    assert (matched, upgraded) == (False, None)


@pytest.mark.parametrize("stored", ["", "not-a-hash", "$argon2id$broken"])
def test_verify_and_upgrade_survives_a_corrupt_stored_hash(stored: str):
    assert verify_and_upgrade("anything", stored) == (False, None)
