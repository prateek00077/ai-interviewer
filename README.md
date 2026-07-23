# AI Interviewer

A multi-tenant AI interviewer. A candidate uploads a resume, joins a voice call
with an AI interviewer, and a recruiter receives a scored report with the
evidence behind every number.

FastAPI · Postgres (RLS) · Redis · Celery · S3/MinIO · NVIDIA Nemotron NIM ·
Pipecat WebRTC

---

# Setup

Two ways to run it. **Path A** puts everything in Docker — nothing to install,
slower to iterate on. **Path B** runs the API and worker on your machine against
dockerised datastores — this is the one to use if you are changing code.

Steps 1–3 are shared. Do them first, whichever path you take.

## Step 1 — Get an NVIDIA API key

**What it does:** buys you the interviewer's ears, brain and voice. Speech
recognition, the language model and text-to-speech all come from NVIDIA NIM.
There is no offline fallback — without a key the app starts and the interview
fails the moment a candidate joins.

Sign up at <https://build.nvidia.com> and create a key. It starts `nvapi-`.
The free tier is enough to run interviews end to end.

One key covers all three services.

## Step 2 — Create your `.env`

**What it does:** copies the annotated template. Every value in it works
locally as-is except the API key.

```bash
cd ai-interviewer
cp .env.example .env
```

Now open `.env` and set **one** line:

```bash
NVIDIA_API_KEY=nvapi-your-key-here
```

That is the only edit required to run locally. [Every other variable is
documented below](#environment-variables) — read that section before deploying
anywhere, but skip it for local testing.

## Step 3 — Start the datastores

**What it does:** brings up Postgres (with the pgvector extension for resume
embeddings), Redis (Celery's broker and live session state) and MinIO (S3-
compatible storage for resumes, recordings, webcam frames and report PDFs).

```bash
docker compose up -d
```

Wait for all three to report healthy:

```bash
docker compose ps
```

**Then, once per machine**, create the database roles:

```bash
docker compose exec -T postgres psql -U postgres -d ai_interviewer \
  -v owner_pw="'owner_pw'" -v app_pw="'app_pw'" < scripts/bootstrap_db.sql
```

This creates two roles: `app_owner` (owns the schema, used only by Alembic) and
`app_user` (unprivileged, used by the app so that Postgres row-level security
actually applies to it). `CREATE ROLE` is not a privilege a migration should
hold, which is why this is a script and not part of `alembic upgrade`.

You only ever run this again after `docker compose down -v`.

---

# Path A — everything in Docker

Use this to try the product. No Python install, no system libraries.

## Step A1 — Build and start the API and worker

**What it does:** builds one image serving both roles and starts two containers
from it — `aii-api` on port 8000 and `aii-worker` running Celery.

```bash
docker compose --profile app up -d --build
```

**The first build takes 15–20 minutes** and produces a ~1.7 GB image: the voice
stack pulls in torch and opencv. Later builds are cached and take under a
minute unless `pyproject.toml` changed.

The API and worker sit behind the `app` profile, so a plain `docker compose up`
(Step 3) starts only the datastores. That is intentional — it is what you want
while running the API locally.

## Step A2 — Create the database schema

**What it does:** runs every migration, creating the tables, enums, RLS
policies and the pgvector index.

```bash
docker compose exec api alembic upgrade head
```

Should end with `0012` as the current head.

## Step A3 — Seed a demo tenant (optional)

**What it does:** creates an organisation, an admin, a recruiter, a job and a
candidate, so you can log in without registering anything by hand.

```bash
docker compose exec api python scripts/seed_data.py
```

It prints the login credentials it created.

**Then skip to [Verify it came up](#verify-it-came-up).**

Useful while running this way:

```bash
docker compose logs -f api worker         # follow both
docker compose --profile app up -d --build   # rebuild after a code change
docker compose --profile app stop api worker # stop just these two
```

There is no hot reload inside the image. If you are editing code, use Path B.

---

# Path B — API and worker on your machine

Use this if you are changing code. You get `--reload`, real tracebacks, and a
debugger.

## Step B1 — Install the system libraries

**What it does:** installs the shared libraries two Python packages load at
*runtime* through ctypes. WeasyPrint (report PDFs) needs Pango and Cairo;
OpenCV (proctoring frames, pulled in by pipecat) needs libGL. Missing ones do
not fail at install — they raise `ImportError` naming `libGL.so.1`, which reads
as a broken install rather than a missing dependency.

```bash
# Debian / Ubuntu
sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgl1

# macOS
brew install pango cairo
```

## Step B2 — Create a virtualenv and install the project

**What it does:** installs the app plus the voice stack and the test tooling.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e '.[voice,dev]'
```

Python 3.11 or newer. The `voice` extra is **not** optional to run the app —
`app.main` imports the API router, which reaches pipecat. Without it the API
dies on startup with `ModuleNotFoundError`. It is an extra only so that CI and
auth-only work install in seconds.

Takes 3–5 minutes; torch is the bulk of it.

## Step B3 — Create the database schema

**What it does:** the same migrations as Step A2, run from your shell.

```bash
alembic upgrade head
```

## Step B4 — Seed a demo tenant (optional)

```bash
python scripts/seed_data.py
```

## Step B5 — Start the API

**What it does:** serves the API, the `/dev` console and the WebRTC signalling
endpoint. `--reload` restarts it on every save.

```bash
uvicorn app.main:app --reload
```

## Step B6 — Start the worker, in a second terminal

**What it does:** runs everything that is not a web request — resume parsing
and embedding, question-plan generation, offline transcription, scoring,
proctoring analysis and PDF rendering.

```bash
celery -A app.workers.celery_app:celery_app worker -l info -c 2
```

**The worker is not optional.** Without it nothing after "upload resume"
happens: no parsing, no question plan, no scoring, no reports. The UI sits on
`PENDING` forever and looks like a hang. If something never leaves `PENDING`,
check this terminal first.

Run exactly one worker. Two workers on the same broker compete for tasks, and
if they are running different code the results look intermittent.

---

# Verify it came up

## Step 4 — Check the process and its dependencies

```bash
curl -s localhost:8000/health    # {"status":"ok"} — the process is alive
curl -s localhost:8000/ready     # every dependency, by name
```

`/ready` should return:

```json
{"status":"ready","checks":{"postgres":"ok","redis":"ok","storage":"ok"}}
```

A 503 names the dependency that is down. Fix it before going further — a red
storage check surfaces three steps later as a confusing presigned-upload
failure, not as a storage error.

## Step 5 — Check the NVIDIA endpoints

**What it does:** probes the language model, speech recognition and
text-to-speech separately, and reports latency per service. Run this before
blaming the app for a silent interview.

```bash
python scripts/check_nim.py                        # Path B
docker compose exec api python scripts/check_nim.py  # Path A
```

All three must report OK.

| | |
|---|---|
| Test console | <http://localhost:8000/dev> |
| API docs | <http://localhost:8000/docs> |
| Liveness / readiness | `/health`, `/ready` |
| MinIO console | <http://localhost:9001> (`minioadmin` / `minioadmin`) |

---

# Testing it

## Step 6 — Drive the whole flow in a browser

Open **<http://localhost:8000/dev>**.

One page, one panel per step, in order: register an organisation → create a job
→ invite a candidate → upload a resume → generate the question plan and rubric
→ join the live voice interview → read the score → download the reports. Every
panel shows the raw request and response, so a 422 tells you exactly what was
sent.

Needs microphone permission for the interview step. Served only when
`ENVIRONMENT` is not `production`.

Two things worth knowing while testing:

- After uploading a resume, wait for the resume panel to reach `READY` before
  generating the plan. Parsing and embedding take a few seconds, and a plan
  generated too early is written from the job description alone.
- Plan generation takes 30–60 seconds. It doubles if the grounding check finds
  a question that is not backed by the resume and asks the model to redo it.

## Step 7 — Or drive it without a browser

**What it does:** runs 47 HTTP steps from invite to report and asserts on each
response. Everything except the voice call itself, which needs a real WebRTC
peer.

```bash
python scripts/e2e_walkthrough.py
```

[docs/manual-testing.md](docs/manual-testing.md) has the full sequence, what
each step should return, and what to try breaking.

## Step 8 — Run the test suite

```bash
ruff check . && mypy app
pytest -q -m "not nim"           # ~6 min; needs the datastores from Step 3
pytest -q -m "not integration"   # ~10 s; no Postgres needed
pytest -q                        # everything, INCLUDING real NVIDIA calls
```

There is no default marker filter, so a bare `pytest -q` runs the 15 `nim`
tests and bills them. That is deliberate: those tests catch failures that only
appear against the real endpoints — an embedding written to the wrong side of
the asymmetric model, a distance threshold tuned into the relevant band — and
hiding them behind a flag nobody remembers is how they rot. Use `-m "not nim"`
while iterating.

Integration tests share your Postgres and clean up after themselves. They never
dispatch Celery tasks: `conftest` makes `apply_async` raise, because a suite
that quietly enqueues real jobs also quietly bills real model calls.

---

# Environment variables

Every variable, its default, and what changes if you touch it. Defaults are
what `.env.example` ships and all of them work locally.

## Required

| Variable | Default | What it does |
|---|---|---|
| `NVIDIA_API_KEY` | — | **The only value you must set.** Covers the LLM (REST), ASR and TTS (both gRPC). |

## NVIDIA NIM

| Variable | Default | What it does |
|---|---|---|
| `NIM_PROFILE` | `cloud` | `cloud` pins NVIDIA's hosted endpoints. `local` lets a reachable entry in `config/services.local.yaml` shadow its cloud counterpart, per service — for running a NIM container yourself. |
| `NIM_REQUEST_TIMEOUT_SECS` | `60` | Ceiling for one model call. Plan generation is the long one. |

## App

| Variable | Default | What it does |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` hides `/docs` and `/dev`, and enforces the `JWT_SECRET` check below. |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Bind address. |
| `LOG_LEVEL` | `INFO` | `DEBUG` adds per-request SQL and retrieval scores. |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated. The candidate page is a browser client, so its origin must be here. |

## Auth

| Variable | Default | What it does |
|---|---|---|
| `JWT_SECRET` | `change-me-in-production` | Signs all four token types, each with a separately derived key. **The app refuses to boot in production** while this is the default or under 32 bytes. Generate: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `JWT_ALGORITHM` | `HS256` | |
| `ACCESS_TOKEN_TTL_MINUTES` | `30` | Recruiter session length. |
| `REFRESH_TOKEN_TTL_DAYS` | `14` | |
| `INTERVIEW_TOKEN_TTL_MINUTES` | `10` | Deliberately short: a leaked interview link must not be replayable later. |
| `INVITE_TTL_HOURS` | `72` | How long a candidate has to accept. |
| `INVITE_MAX_REDEMPTIONS` | `3` | Multi-use so a candidate whose browser crashes can rejoin. Each redemption still yields only a 10-minute interview token. |

## Database

| Variable | Default | What it does |
|---|---|---|
| `DATABASE_URL` | `...app_user:app_pw@localhost:5432/...` | The app's connection, as an unprivileged role that owns nothing — which is what makes row-level security apply to it. |
| `DATABASE_OWNER_URL` | `...app_owner:owner_pw@localhost:5432/...` | **Alembic only.** Owns the schema; the app must never use it. |

Both are overridden inside the containers to reach `postgres` rather than
`localhost` — see `docker-compose.yml`. Change the passwords here and you must
also change them in the `bootstrap_db.sql` invocation in Step 3.

## Redis

| Variable | Default | What it does |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Live interview checkpoints and rate-limit counters. |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Task queue. |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/2` | Task results — the post-interview chain needs these to pass data between steps. |

Three separate databases on one Redis so that flushing one does not disturb the
others.

## Object storage

| Variable | Default | What it does |
|---|---|---|
| `S3_ENDPOINT_URL` | `http://localhost:9000` | MinIO locally; leave unset for real AWS S3. |
| `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | `minioadmin` | Match the MinIO credentials in `docker-compose.yml`. |
| `S3_BUCKET_RESUMES` | `resumes` | Uploaded CVs. |
| `S3_BUCKET_RECORDINGS` | `recordings` | Interview audio, written when the session ends. |
| `S3_BUCKET_PROCTORING` | `proctoring` | Webcam frames. |
| `S3_BUCKET_REPORTS` | `reports` | Rendered PDFs. |

The buckets are created automatically on first start by the `minio-init`
service.

## Voice

| Variable | Default | What it does |
|---|---|---|
| `CONNECT_PREWARM_TIMEOUT_SECS` | `45` | Session start waits for a TTS warm-up. Magpie's cold start is real, and without this the candidate's first question is silence. |
| `VAD_STOP_SECS` | `0.8` | Silence before the candidate is treated as finished. **0.2 cuts off anyone who pauses to think mid-answer** — raise it if the interviewer interrupts. |
| `VAD_MIN_VOLUME` | `0.4` | Speech threshold. Lower it in a noisy room only if speech is being missed. |
| `VOICE_IDLE_NUDGE_SECS` | `25` | Silence before the interviewer checks in ("still with me?"). |
| `VOICE_MAX_IDLE_NUDGES` | `2` | Nudges before it says goodbye and ends the session. |
| `MAX_INTERVIEW_MINUTES` | `45` | Hard cap. The session ends whatever is happening. |

## Proctoring

Per-job policy overrides all of these; these are the defaults a new job gets.

| Variable | Default | What it does |
|---|---|---|
| `PROCTOR_BLUR_LIMIT` | `3` | Tab-switches before it is flagged. |
| `PROCTOR_AUTO_TERMINATE` | `false` | Whether a breach ends the interview. Off by default — a false positive that ejects a candidate mid-answer is worse than a flag on a report. |
| `PROCTOR_CAMERA_ENABLED` | `true` | Whether webcam frames are captured at all. |
| `PROCTOR_FRAME_INTERVAL_SECS` | `10` | Seconds between frames. |

## Rate limiting

| Variable | Default | What it does |
|---|---|---|
| `REGISTER_MAX_ATTEMPTS` | `5`/hour/IP | Commented out in `.env.example`. |
| `LOGIN_MAX_ATTEMPTS` | `10` | Commented out. |

Those are the **production** numbers. In development both are raised
automatically (500 and 1000) because five registrations an hour makes the
product untestable — every walkthrough run needs a fresh tenant. Set either
explicitly and your value wins everywhere, including a deliberately low one to
exercise the 429 path.

If you do hit a 429 locally, clear the counters:

```bash
docker exec aii-redis redis-cli --scan --pattern 'rl:*' | xargs docker exec -i aii-redis redis-cli DEL
```

## Email

| Variable | Default | What it does |
|---|---|---|
| `SMTP_HOST` | *(empty)* | **Empty disables sending.** Invites still work — the token comes back in the API response, and the `/dev` console uses it directly. |
| `SMTP_PORT` | `587` | |
| `SMTP_USER` / `SMTP_PASSWORD` | *(empty)* | |
| `SMTP_STARTTLS` | `true` | |
| `EMAIL_FROM` | `noreply@example.com` | |
| `APP_BASE_URL` | `http://localhost:3000` | Where the candidate app lives. Invite links are built from this and **never** from a request `Host` header, which an attacker controls. |

## Reports

| Variable | Default | What it does |
|---|---|---|
| `REPORT_DOWNLOAD_TTL_SECS` | `3600` | Lifetime of a presigned report URL. |

## Not in `.env.example`

These are read from the environment like any other setting, but are left out of
the template because their defaults are right for almost everyone. Set them the
same way if you need to.

| Variable | Default | What it does |
|---|---|---|
| `JWT_ISSUER` | `aii` | `iss` claim. Change it and existing tokens stop validating. |
| `LOGIN_ATTEMPT_WINDOW_SECONDS` | `900` | Window the login limit is counted over. |
| `REGISTER_ATTEMPT_WINDOW_SECONDS` | `3600` | Same, for registration. |
| `DEV_LOGIN_MAX_ATTEMPTS` | `1000` | The relaxed development limit. Applies only when `ENVIRONMENT` is not `production` and `LOGIN_MAX_ATTEMPTS` is unset. |
| `DEV_REGISTER_MAX_ATTEMPTS` | `500` | Same, for registration. |
| `DB_POOL_SIZE` | `10` | Pooled connections. |
| `DB_MAX_OVERFLOW` | `5` | Extra connections under load. |
| `WEBRTC_STUN_URLS` | `[]` | STUN servers for the interview call. Empty works on a LAN; set it for candidates behind NAT. |
| `INTERVIEW_EXPIRY_HOURS` | `72` | After this an unstarted interview is reaped. |
| `MAX_RESUME_BYTES` | `10485760` | 10 MB upload ceiling, enforced when the presigned URL is issued. |
| `MAX_PROCTOR_FRAME_BYTES` | `2097152` | 2 MB per webcam frame. |
| `PROCTOR_EVENTS_PER_MINUTE` | `120` | Per-socket cap. A candidate controls that socket, so it is rate-limited. |
| `S3_REGION` | `us-east-1` | Ignored by MinIO; matters on real S3. |
| `S3_PRESIGN_TTL_SECS` | `900` | Lifetime of an upload URL. |
| `SMTP_TIMEOUT_SECS` | `15` | A slow mail server must not hold up an invite. |

---

# Resetting

**Restart the app only** — keeps all data:

```bash
docker compose --profile app restart api worker    # Path A
# Path B: Ctrl-C both terminals, start them again
```

**Wipe everything** — deletes the database and all of MinIO:

```bash
docker compose --profile app down
docker compose down -v
```

Then start again from **Step 3**, including `bootstrap_db.sql` — dropping the
volumes drops the roles it created.

---

# Troubleshooting

| Symptom | Cause |
|---|---|
| Anything stuck on `PENDING` | The worker is not running, or crashed. Check its terminal. |
| `/dev` returns 500 | `static/console.html` is missing from the image. Rebuild. |
| `/ready` says `503` | It names the failing dependency. Start with that one. |
| Interview connects but nobody speaks | Run `scripts/check_nim.py`. Usually an invalid or rate-limited API key. |
| Interviewer interrupts mid-answer | Raise `VAD_STOP_SECS`. |
| `ModuleNotFoundError: pipecat` | Installed without the `voice` extra. See Step B2. |
| `ImportError: libGL.so.1` | Missing system libraries. See Step B1. |
| `429` when registering | Rate limit. See the flush command above. |
| Plan questions ignore the resume | The plan was generated before the resume reached `READY`. Regenerate. |
| Results change between runs | More than one Celery worker on the same broker. `pgrep -af celery` and kill the extras. |

---

# Docs

- [Architecture](docs/architecture.md) — how it fits together, and why
- [API reference](docs/api.md) — who may call what
- [Manual testing](docs/manual-testing.md) — driving it by hand, and what to try breaking
- [ADR-0001](docs/adr/0001-fastapi-over-fastify.md) — FastAPI over Fastify
- [ADR-0002](docs/adr/0002-single-process-voice.md) — voice in-process, and its debt
