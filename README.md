# AI Interviewer

A multi-tenant AI interviewer. A candidate uploads a resume, joins a voice call
with an AI interviewer, and a recruiter receives a scored report with the
evidence behind every number.

FastAPI · Postgres (RLS) · Redis · Celery · S3/MinIO · NVIDIA Nemotron NIM ·
Pipecat WebRTC

## Run it

```bash
docker compose up -d                      # postgres + pgvector, redis, minio

# once: create the database roles (CREATE ROLE is not Alembic's to hold)
docker compose exec -T postgres psql -U postgres -d ai_interviewer \
  -v owner_pw="'owner_pw'" -v app_pw="'app_pw'" < scripts/bootstrap_db.sql

alembic upgrade head
python scripts/seed_data.py

uvicorn app.main:app --reload                                   # API
celery -A app.workers.celery_app:celery_app worker -l info      # worker
```

The worker is not optional. Without it nothing past "upload resume" happens:
no parsing, no question plan, no scoring, no reports.

| | |
|---|---|
| Test console | <http://localhost:8000/dev> |
| API docs | <http://localhost:8000/docs> |
| Liveness / readiness | `/health`, `/ready` |
| MinIO console | <http://localhost:9001> (`minioadmin` / `minioadmin`) |

## Docs

- [Architecture](docs/architecture.md) — how it fits together, and why
- [API reference](docs/api.md) — who may call what
- [Manual testing](docs/manual-testing.md) — driving it by hand, and what to try breaking
- [ADR-0001](docs/adr/0001-fastapi-over-fastify.md) — FastAPI over Fastify
- [ADR-0002](docs/adr/0002-single-process-voice.md) — voice in-process, and its debt

## Checks

```bash
ruff check . && mypy app && pytest -q
python scripts/check_nim.py          # probe the three NVIDIA endpoints
python scripts/e2e_walkthrough.py    # full flow against a running server
```
