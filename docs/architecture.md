# Architecture

A multi-tenant AI interviewer. A candidate uploads a resume, joins a voice call
with an AI interviewer, and a recruiter receives a scored report with the
evidence behind every number.

Two deviations from the reference design, both recorded as ADRs: the backend is
**FastAPI + Celery** rather than Fastify + BullMQ
([ADR-0001](adr/0001-fastapi-over-fastify.md)), and the voice stack is **NVIDIA
Nemotron NIM** rather than Whisper + Qwen.

## The shape of it

```
                        ┌──────────────────────────────┐
  browser ── WebRTC ───▶│  voice/  (in-process, ADR-2) │
     │                  │  Pipecat: ASR → LLM → TTS    │
     │                  └───────────┬──────────────────┘
     │                              │ core/events (in-process bus)
     │                              ▼
     ├── HTTPS ────▶ api/v1 ──▶ modules/{auth,users,jobs,resume,
     │                          question_plan,interview,proctoring,
     │                          scoring,reports}
     │                              │
     └── WS ───────▶ proctoring     ▼
                              Postgres (RLS)  ·  Redis  ·  S3/MinIO
                                    │
                              Celery workers ──▶ NIM (LLM, embeddings, vision)
```

### Layering rule

`api/v1` routes are thin: validate, delegate, shape a response. Every decision
that matters lives in `modules/`, where it is testable without HTTP. A route
containing an `if` about business state is in the wrong place.

## Tenancy is enforced by Postgres, not by application code

Every org-scoped table has `FORCE ROW LEVEL SECURITY` and a generated policy
(`db/rls.py`). Policies read three GUCs — `app.current_org`, `app.actor_kind`,
`app.actor_id` — set per transaction from the bearer token's **signed claims**,
never from a header, path, query or body.

The consequence worth internalising: a route that forgets `WHERE org_id = ...`
is still safe. The policy filters for it.

`FORCE` is what subjects the table owner to its own policies. Without it a
misconfigured `DATABASE_URL` pointing at the owner role sees everything, and
the isolation tests pass while proving nothing.

Policy shapes, dispatched in `rls.policy_for`:

| Shape | Applies to | Effect |
|---|---|---|
| org root | `organizations` | matches on `id`, not `org_id` |
| tenant | most tables | org isolation only |
| user-only | plans, scores, proctoring, recruiter reports | candidates read nothing |
| candidate-scoped | interviews, candidates, candidate reports | candidates read own |
| candidate-writable | `resumes` | candidates read *and* write own |

Adding a tenant table is a one-line registry change in `db/base.py`; the
migration loop generates the policy, and a coverage-guard test fails if you
forget.

**The one deliberate bypass** is login: an email with no org yet is a chicken
and egg. Rather than a hole in the users policy, a `SECURITY DEFINER` function
returns six columns for one email, owned by `app_auth` — a NOLOGIN role that
exists for that function and is named by exactly one policy.

### Three database roles

| Role | Holds | Used by |
|---|---|---|
| `app_owner` | DDL | Alembic |
| `app_user` | DML only, owns nothing | the application |
| `app_auth` | `SELECT` on two tables, NOLOGIN | one definer function |

## Authentication

Four token types signed with **four different keys**, each derived from one
`JWT_SECRET` by domain-separated HMAC-SHA256. Presenting an invite token where
an access token belongs fails at *signature verification*, before any claim is
read — a structural failure rather than a forgotten `if`. A distinct `aud` per
type and an explicit `typ` assertion sit on top.

| Token | TTL | Holder |
|---|---|---|
| access | 30 min | recruiter |
| refresh | 14 days | recruiter |
| invite | 72 h | candidate (multi-use link) |
| interview | 10 min | candidate (session credential) |

Candidates hold no credentials — there is no password column on `candidates`.
Passwords are Argon2id via pwdlib, with a dummy-hash verify on the unknown-user
path so login timing does not reveal whether an address is registered. Refresh
tokens rotate with reuse detection, decided inside one Redis Lua script so
check-and-consume is atomic; reuse tombstones the whole family.

Access tokens are stateless, so one stays valid up to its full 30-minute TTL
after logout. `core/dependencies.py` names the single place a Redis denylist
would go if that window ever becomes unacceptable.

## The voice module is isolated behind an event bus

`modules/voice/` runs in the API process
([ADR-0002](adr/0002-single-process-voice.md)). Its only outward channel is
`core/events` — a fire-and-forget in-process bus carrying `SessionStarted`,
`SessionEnded`, `TurnCompleted`, `ProctorEventRaised`. Nothing outside imports
`voice/` except through `session_manager`, and `voice/` imports nothing from the
rest.

That boundary is what lets the module move to its own process later without
touching either side. It also enforces a rule: **the voice module never writes
an interview status.** It announces what happened; `interview/service` decides
what that means. Every status change goes through `interview/state_machine`,
which has an explicit legal-transition table and terminal states with no
outgoing edges.

## The post-interview pipeline

Enqueued when the session ends, once per interview ending rather than once per
event delivery. Every link takes `(org_id, interview_id)` and uses an immutable
signature, so each is independently re-runnable by hand.

```
finalize → correct_transcript → [measure_signals ∥ analyze_frames]
         → score_interview → finalize_verdict
         → [render_recruiter ∥ render_candidate]
```

The order is a dependency graph, not a preference: the scorer verifies its
quotes against the transcript, so correction runs first; the verdict is
recomputed from the full event set including what the vision pass writes; both
reports print results settled by everything above them.

Tasks are at-least-once and idempotent on `interview_id`.

### Transcription

The NVCF ASR function is **online-only** — `offline_recognize` is an error
against it, not a slower path. The "offline" pass therefore streams the recorded
file through the *streaming* API. Riva does not require real-time pacing, so a
30-minute call decodes in well under a minute.

It corrects candidate turns only. Interviewer text is what was sent to TTS and
is already exact; re-transcribing our own synthesised speech can only add
errors. Timings are never touched — the live session stamped them against the
audio clock.

## Scoring, and what it deliberately does not do

- **One model call per criterion.** Batching lets a strong first answer colour
  the rest.
- **Every quote is verified against the transcript.** Paraphrases and
  fabrications are dropped; a score with no surviving evidence is discarded. An
  unsupported number still reads as authoritative and makes a rejection
  unappealable.
- **Weights renormalise over the graded subset.** Three criteria worth 0.6, all
  scored 4.0, must not report 2.4 — that reads as a mediocre candidate when we
  simply never asked 40% of the questions. Below 50% coverage: no number at all.
- **`INSUFFICIENT_EVIDENCE` ≠ `NO_HIRE`.** Someone whose audio failed has not
  been assessed.
- **Delivery signals are multiplied by nothing.** Pitch variance, pauses and
  filler rate track anxiety far better than competence. Measured, reported,
  never scored — and `aggregator` does not import `confidence`, which a test
  asserts.

## The two reports

The candidate must never learn their score. Enforced at five layers, each
independently sufficient:

1. `candidate_reports` has no score column.
2. `CandidateView` has no score attribute — and `slots=True`, so one cannot be
   attached at runtime.
3. `candidate.generate` takes `(job_title, topic_names, turns)`. No
   score-bearing parameter exists to pass one to.
4. The candidate template references no score variable and includes no shared
   partial.
5. `CandidateFeedbackRead` has no score field and shares no base class with the
   recruiter schema.

Two tables rather than one with an `audience` column, because with a
discriminator the only thing between a candidate and their hire recommendation
is a `WHERE` clause somebody has to remember, every time, forever.

The recruiter report shows every number next to what produced it: a score with
its evidence and the offset in the recording, an overall with its rubric
coverage, a verdict with its reasons.

## Operational endpoints

| Endpoint | Question | On failure |
|---|---|---|
| `/health` | is the process alive? | orchestrator restarts |
| `/ready` | can it serve traffic? | pulled from the load balancer |
| `/metrics` | what to page on | — |

Separate because conflating them is how a deployment takes itself down:
`/health` wired to Postgres means a database blip restarts every pod at once,
against a database already struggling.

Shutdown drains: new voice sessions are refused (409) before anything closes,
while sessions already running finish — each is a real person mid-sentence.

## Where things live

```
app/
  api/{ops,deps}.py, api/v1/*      routes; thin
  core/                            config, security, events, exceptions, logging
  db/                              base (registries), session (tenant GUCs), rls
  models/                          SQLAlchemy; one module per aggregate
  modules/                         the actual logic
  integrations/                    storage (S3), nim_client, email
  workers/                         celery_app, pipeline, tasks/
config/
  services.{cloud,local}.yaml      NIM endpoints; no code branches on provider
  prompts/*.yaml                   edited far more often than the code around them
  templates/*.html                 one per report audience, no shared partials
```

## Running it

```bash
docker compose up -d                 # postgres, redis, minio
docker compose exec -T postgres psql -U postgres -d ai_interviewer \
  -v owner_pw="'owner_pw'" -v app_pw="'app_pw'" < scripts/bootstrap_db.sql
alembic upgrade head
python scripts/seed_data.py
uvicorn app.main:app --reload
celery -A app.workers.celery_app:celery_app worker --loglevel=info

docker compose --profile app up -d   # or: everything containerised
```

`python scripts/check_nim.py` probes all three NIM services and prints
pass/fail with measured latency — run it first when anything voice-related
misbehaves.
