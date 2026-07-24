"""Application settings via pydantic-settings (env-driven)."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_JWT_SECRET = "change-me-in-production"
MIN_JWT_SECRET_BYTES = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    environment: Literal["development", "staging", "production", "test"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    # NoDecode: without it pydantic-settings JSON-parses complex types straight
    # from the dotenv value, before the validator below ever sees the string.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # --- Auth ---
    jwt_secret: SecretStr = SecretStr(DEFAULT_JWT_SECRET)
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "aii"
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_days: int = 14
    interview_token_ttl_minutes: int = 10
    invite_ttl_hours: int = 72
    invite_max_redemptions: int = 3

    # --- Rate limiting ---
    #
    # These are the PRODUCTION numbers. ``_relax_rate_limits_in_development``
    # below raises them for local work unless you set them yourself: five
    # registrations an hour is correct for a public endpoint and makes the
    # product untestable on a laptop, where every run of the walkthrough needs a
    # fresh tenant.
    #
    # Raised, not disabled. A limiter that does not exist in development is one
    # nobody notices is broken until production, and the 429 path -- the header,
    # the error shape, the retry_after -- should still be reachable by hammering.
    login_max_attempts: int = 10
    login_attempt_window_seconds: int = 900
    register_max_attempts: int = 5
    register_attempt_window_seconds: int = 3600

    # What development gets instead. Generous enough that ordinary manual
    # testing never trips it, low enough that a runaway loop still does.
    dev_login_max_attempts: int = 1_000
    dev_register_max_attempts: int = 500

    # --- Database ---
    # The app connects as an unprivileged role so RLS actually applies to it.
    database_url: str = "postgresql+asyncpg://app_user:app_pw@localhost:5432/ai_interviewer"
    # Alembic connects as the schema owner. The app must never use this.
    database_owner_url: str = (
        "postgresql+asyncpg://app_owner:owner_pw@localhost:5432/ai_interviewer"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 5

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Interview ---
    # How long the candidate must be silent before their turn is considered
    # over. MEASURED AS TOO SHORT AT 0.2: someone thinking mid-answer -- "we
    # sharded on... tenant id" -- had the pause read as the end of their turn,
    # so the ASR emitted a fragment and the interviewer replied to half a
    # sentence. An interview is not a chat assistant; people pause to think, and
    # the cost of waiting an extra half second is a beat of dead air, while the
    # cost of cutting in is the answer.
    vad_stop_secs: float = 0.8
    # Below this a frame is silence regardless of Silero's confidence. Lowered
    # from pipecat's 0.6 default because a softly-spoken candidate on a laptop
    # microphone was being treated as background noise.
    vad_min_volume: float = 0.4

    # --- Idle handling ---
    # Silence long enough that the candidate has probably lost audio, walked
    # away, or is waiting for a prompt they will never get. The interviewer
    # checks in rather than sitting mute.
    voice_idle_nudge_secs: float = 25.0
    # How many times to check in before concluding nobody is there. Two nudges
    # then end: a third would be badgering someone who has clearly gone.
    voice_max_idle_nudges: int = 2
    # Magpie's cold start is real (687ms measured). Session start waits for a
    # warm TTS rather than opening with dead air.
    connect_prewarm_timeout_secs: float = 45.0
    # STUN servers for WebRTC NAT traversal. Empty means host candidates only,
    # which works locally and not across the internet. TURN is still needed for
    # candidates behind a symmetric NAT; see docs before deploying.
    webrtc_stun_urls: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # A hard cap on one session. Protects the turn budget, the API bill, and a
    # candidate who walked away from their laptop with the mic open.
    max_interview_minutes: int = 45
    # How long an INVITED interview may sit before the reaper expires it.
    interview_expiry_hours: int = 72

    # --- Proctoring (per-job policy overrides these) ---
    proctor_blur_limit: int = 3
    proctor_auto_terminate: bool = False
    proctor_camera_enabled: bool = True
    proctor_frame_interval_secs: int = 10
    # A webcam still, not a document. Anything larger is a mistake or an attack.
    max_proctor_frame_bytes: int = 2 * 1024 * 1024
    # A candidate controls the proctoring socket completely, so the ingest rate
    # is bounded regardless of what the browser claims to be reporting.
    proctor_events_per_minute: int = 120

    # --- Object storage (MinIO locally, R2/S3 in production) ---
    # Empty means "AWS S3 proper"; boto3 then resolves the regional endpoint.
    s3_endpoint_url: str | None = "http://localhost:9000"
    # The host the *browser* must reach, baked into presigned URLs. It differs
    # from s3_endpoint_url exactly when the API runs somewhere the browser
    # cannot: inside Docker, the API talks to MinIO at "minio:9000", but a
    # presigned URL carrying that host is unreachable from the host machine and
    # the PUT dies as "TypeError: Failed to fetch". Left empty, presigning falls
    # back to s3_endpoint_url, which is correct for a host-local API.
    s3_public_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key_id: SecretStr = SecretStr("minioadmin")
    s3_secret_access_key: SecretStr = SecretStr("minioadmin")
    s3_bucket_resumes: str = "resumes"
    s3_bucket_recordings: str = "recordings"
    s3_bucket_proctoring: str = "proctoring"
    s3_bucket_reports: str = "reports"
    # Short: a presigned PUT is handed to a browser that is about to use it.
    s3_presign_ttl_secs: int = 900
    # Resumes are documents, not media. Anything larger is a mistake or an attack.
    max_resume_bytes: int = 10 * 1024 * 1024

    # --- Celery ---
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # --- NVIDIA NIM ---
    # One key covers LLM (REST), ASR (gRPC) and TTS (gRPC).
    nvidia_api_key: SecretStr = SecretStr("")
    # "local" lets a reachable services.local.yaml entry shadow its cloud counterpart.
    # "cloud" pins the hosted endpoints even when a local NIM happens to be up.
    nim_profile: Literal["cloud", "local"] = "cloud"
    nim_request_timeout_secs: float = 60.0

    # --- Email ---
    # An empty host disables sending. Deliberately the default: local
    # development must not need a mail server, and a misconfigured deployment
    # should log the message it would have sent rather than crash the invite.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: SecretStr = SecretStr("")
    smtp_starttls: bool = True
    email_from: str = "noreply@example.com"
    smtp_timeout_secs: float = 15.0

    # Where the candidate-facing app lives. Invite emails are the only place a
    # URL is constructed rather than received, and it cannot come from a request
    # header -- a Host header is attacker-controlled, and using one here would
    # let anyone mint an invite email pointing at their own domain.
    app_base_url: str = "http://localhost:3000"

    # --- Reports ---
    # Longer than the upload presign: a recruiter opens a report, reads it, and
    # comes back to it. Still bounded, because the URL is a bearer credential
    # for a document containing an assessment of a named person.
    report_download_ttl_secs: int = 3600

    @model_validator(mode="after")
    def _relax_rate_limits_in_development(self) -> "Settings":
        """Raise the auth limits for local work, unless they were set explicitly.

        ``model_fields_set`` is what makes this safe: it holds only the fields
        that actually came from the environment or the constructor, so anyone
        who sets REGISTER_MAX_ATTEMPTS keeps their value -- including someone
        setting it low on purpose to exercise the 429 path.

        Development only. Staging and production keep the real numbers, because
        a limit that quietly evaporates on the way to production is worse than
        no limit at all: you would ship believing you had one.
        """
        if self.environment != "development":
            return self

        if "register_max_attempts" not in self.model_fields_set:
            self.register_max_attempts = self.dev_register_max_attempts
        if "login_max_attempts" not in self.model_fields_set:
            self.login_max_attempts = self.dev_login_max_attempts
        return self

    @field_validator("cors_origins", "webrtc_stun_urls", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("jwt_algorithm")
    @classmethod
    def _hmac_only(cls, v: str) -> str:
        # The per-type key derivation in core.security produces symmetric keys, so an
        # asymmetric algorithm here would fail at signing time in a confusing way.
        if not v.startswith("HS"):
            raise ValueError(f"jwt_algorithm must be an HMAC algorithm (HS*), got {v!r}")
        return v

    @model_validator(mode="after")
    def _reject_weak_secret_in_production(self) -> "Settings":
        if self.environment != "production":
            return self
        secret = self.jwt_secret.get_secret_value()
        if secret == DEFAULT_JWT_SECRET:
            raise ValueError(
                "JWT_SECRET is still the example default; refusing to boot in production"
            )
        if len(secret.encode()) < MIN_JWT_SECRET_BYTES:
            raise ValueError(
                f"JWT_SECRET must be at least {MIN_JWT_SECRET_BYTES} bytes in production"
            )
        # A missing key only shows up when the first candidate joins an interview and
        # the pipeline fails to build. Fail at boot instead.
        if self.nim_profile == "cloud" and not self.nvidia_api_key.get_secret_value():
            raise ValueError("NVIDIA_API_KEY is required when NIM_PROFILE=cloud")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Tests override this rather than mutating the module global."""
    return Settings()


settings = get_settings()
