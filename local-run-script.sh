#!/usr/bin/env bash
#
# LOCAL-ENVIRONMENT run for AI Interviewer (README Path B).
#
# The API and the Celery worker run on THIS machine, in a Python venv, with the
# reload loop -- the setup you want while editing code. Only the things that are
# awkward to run on a host -- Postgres, Redis, MinIO -- run in Docker. For the
# other way (everything in Docker, nothing but Docker installed) use its sibling,
# ./docker-run-script.sh -- see README Path A.
#
# Run it:
#
#   ./local-run-script.sh
#
# It brings the datastores up, creates the venv the first time, migrates, seeds,
# starts the worker in the background, then runs the API in the foreground.
# Press Ctrl-C to stop: the worker is stopped with it.
#
# The one thing you must do by hand is put a real NVIDIA API key in .env
# (https://build.nvidia.com -- it starts `nvapi-`). This script refuses to go
# further without it.
#
# Safe to re-run: every step is idempotent, so this also serves as "bring it
# back up after a reboot". To wipe the datastores and start clean:
#
#   docker compose down -v
#
# ...then run this again (it re-creates the DB roles that the volumes hold).

set -euo pipefail

# Always operate from the repo root (this script's own directory).
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"
RUN_DIR=".local-run"          # worker log + pid live here; gitignored
WORKER_LOG="$RUN_DIR/worker.log"

# --- pretty output ----------------------------------------------------------
bold=$'\033[1m'; green=$'\033[32m'; yellow=$'\033[33m'; red=$'\033[31m'; reset=$'\033[0m'
step()  { printf '\n%s==> %s%s\n' "$bold" "$1" "$reset"; }
info()  { printf '    %s\n' "$1"; }
ok()    { printf '    %s%s%s\n' "$green" "$1" "$reset"; }
warn()  { printf '    %s%s%s\n' "$yellow" "$1" "$reset"; }
die()   { printf '\n%sERROR: %s%s\n' "$red" "$1" "$reset" >&2; exit 1; }

# --- 1. preflight (fail fast) ----------------------------------------------
step "Checking prerequisites"

command -v docker >/dev/null 2>&1 \
  || die "Docker is not installed or not on PATH. The datastores run in Docker even on this path."
docker info >/dev/null 2>&1 \
  || die "The Docker daemon is not reachable. Start Docker and retry."
docker compose version >/dev/null 2>&1 \
  || die "Docker Compose v2 is unavailable ('docker compose'). Update Docker and retry."
command -v python3 >/dev/null 2>&1 \
  || die "python3 is not installed or not on PATH. This path runs the app on your host, so it needs Python 3.11+."

py_ok="$(python3 -c 'import sys; print(1 if sys.version_info[:2] >= (3, 11) else 0)' 2>/dev/null || echo 0)"
[[ "$py_ok" == "1" ]] \
  || die "Python 3.11+ is required (found $(python3 -V 2>&1)). Install a newer Python and retry."
ok "Docker, Docker Compose and Python 3.11+ are available."

# Port 8000 must be free -- if the all-in-Docker path is up, its API owns it.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx aii-api; then
  die "The all-in-Docker API (aii-api) is running and owns port 8000.
    Stop just the app containers, keeping the datastores, then re-run this:
        docker compose --profile app stop api worker"
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  die ".env did not exist, so I created it from .env.example.
    Now open .env and set NVIDIA_API_KEY to your key (get one at https://build.nvidia.com,
    it starts 'nvapi-'), then re-run ./local-run-script.sh."
fi

# The key must be present and not the shipped placeholder.
key_line="$(grep -E '^[[:space:]]*NVIDIA_API_KEY=' .env | tail -n1 || true)"
key_val="${key_line#*=}"
if [[ -z "$key_val" || "$key_val" == nvapi-xxxx* || "$key_val" != nvapi-* ]]; then
  die "NVIDIA_API_KEY in .env is missing or still the placeholder.
    Get a key at https://build.nvidia.com (it starts 'nvapi-'), set this line in .env:
        NVIDIA_API_KEY=nvapi-your-key-here
    then re-run ./local-run-script.sh. There is no offline fallback -- the interview needs it."
fi
ok "NVIDIA_API_KEY is set."

# On this path the app reads .env directly, so its URLs must point at localhost
# (the datastores are reached over the host's mapped ports). The all-in-Docker
# path overrides these to container names, which is why the same .env serves both
# -- but only if what is written here is the localhost form.
db_url="$(grep -E '^[[:space:]]*DATABASE_URL=' .env | tail -n1 || true)"
if [[ -n "$db_url" && "$db_url" != *localhost* && "$db_url" != *127.0.0.1* ]]; then
  warn "DATABASE_URL in .env does not point at localhost:"
  warn "    ${db_url}"
  warn "On this path the app runs on your host, so it must reach Postgres at localhost:5432."
  warn "If you edited .env for the Docker path, restore the localhost form from .env.example."
fi

# --- 2. Python venv (create + install once) --------------------------------
step "Preparing the Python environment"
if [[ ! -x "$PY" ]]; then
  info "No venv yet -- creating $VENV and installing dependencies."
  warn "First install pulls torch + opencv and takes a few minutes. Later runs skip this."
  python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip >/dev/null
  # Same extras the container installs: voice (pipecat + nvidia) and nim (Riva
  # client for check_nim). Without voice, `app.main` dies on import.
  "$PY" -m pip install -e '.[voice,nim]'
  ok "Virtualenv created and dependencies installed."
else
  ok "Reusing existing venv at $VENV."
fi

# --- 3. datastores up (Docker) ---------------------------------------------
step "Starting datastores (Postgres, Redis, MinIO)"
# No --profile app: this brings up ONLY the backing services, not the containerised
# API/worker. Those run on your host, below.
docker compose up -d

info "Waiting for Postgres, Redis and MinIO to report healthy..."
deadline=$(( SECONDS + 120 ))
while :; do
  # Named explicitly so this stays correct regardless of whether the Path A
  # app containers happen to be up.
  unhealthy="$(docker compose ps postgres redis minio --format '{{.Service}} {{.Health}}' \
    | awk '$2 != "" && $2 != "healthy" { print $1 }' || true)"
  [[ -z "$unhealthy" ]] && break
  if (( SECONDS >= deadline )); then
    docker compose ps
    die "Timed out waiting for datastores to become healthy: ${unhealthy//$'\n'/, }"
  fi
  sleep 3
done
ok "Datastores are healthy."

# --- 4. bootstrap DB roles (idempotent) ------------------------------------
step "Bootstrapping database roles and pgvector"
docker compose exec -T postgres psql -U postgres -d ai_interviewer \
  -v owner_pw="'owner_pw'" -v app_pw="'app_pw'" < scripts/bootstrap_db.sql
ok "Roles (app_owner, app_user, app_auth) and the vector extension are in place."

# --- 5. migrate schema (on the host) ---------------------------------------
step "Applying database migrations"
# Alembic reads DATABASE_OWNER_URL from .env, which points at localhost here.
"$VENV/bin/alembic" upgrade head
ok "Schema is at head (expected 0012)."

# --- 6. seed demo tenant ----------------------------------------------------
step "Seeding a demo tenant"
info "Login credentials are printed below by seed_data.py -- keep them handy:"
"$PY" scripts/seed_data.py
ok "Demo org, recruiter and candidate created."

# --- 7. start the worker (background) ---------------------------------------
step "Starting the Celery worker (background)"
mkdir -p "$RUN_DIR"
worker_pid=""
tail_running=""

# Stop the worker whenever this script exits, however it exits. Declared before
# the worker starts so an early failure still cleans up.
cleanup() {
  trap - EXIT INT TERM
  [[ -n "$worker_pid" ]] && kill "$worker_pid" 2>/dev/null || true
  [[ -n "$worker_pid" ]] && wait "$worker_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --concurrency=2 matches the container: these tasks are long and memory-hungry,
# and the ceiling is the shared NIM rate limit, not local cores.
"$VENV/bin/celery" -A app.workers.celery_app:celery_app worker \
  --loglevel=info --concurrency=2 >"$WORKER_LOG" 2>&1 &
worker_pid=$!

info "Waiting for the worker to come up (log: $WORKER_LOG)..."
deadline=$(( SECONDS + 60 ))
while :; do
  if ! kill -0 "$worker_pid" 2>/dev/null; then
    warn "Worker exited early. Last lines of $WORKER_LOG:"
    tail -n 20 "$WORKER_LOG" >&2 || true
    die "The Celery worker failed to start. See $WORKER_LOG above."
  fi
  # Celery prints 'celery@<host> ready.' once the broker connection is live.
  if grep -q "ready\." "$WORKER_LOG" 2>/dev/null; then
    break
  fi
  if (( SECONDS >= deadline )); then
    die "Worker did not report ready within 60s. See $WORKER_LOG."
  fi
  sleep 1
done
ok "Worker is ready (pid $worker_pid)."

# --- 8. NVIDIA check (warning only) ----------------------------------------
step "Checking NVIDIA NIM endpoints"
if "$PY" scripts/check_nim.py >/dev/null 2>&1; then
  ok "NVIDIA NIM endpoints (LLM, ASR, TTS) all responded."
else
  warn "NVIDIA NIM check failed -- interviews will be silent until it passes."
  warn "Run '$PY scripts/check_nim.py' to see which service and why (usually a bad or rate-limited key)."
fi

# --- 9. banner + run the API in the foreground -----------------------------
printf '\n%s==================================================================%s\n' "$green" "$reset"
printf '%s  AI Interviewer -- API starting in the foreground below.%s\n' "$bold" "$reset"
printf '\n      %s%shttp://localhost:8000/dev%s   <- the test console\n' "$bold" "$green" "$reset"
printf '\n  Other URLs:\n'
printf '      API docs       http://localhost:8000/docs\n'
printf '      MinIO console  http://localhost:9001  (minioadmin / minioadmin)\n'
printf '\n  Running here:\n'
printf '      API      this terminal, with --reload (edit code, it restarts)\n'
printf '      Worker   background, pid %s, logs in %s\n' "$worker_pid" "$WORKER_LOG"
printf '      Data     Postgres, Redis, MinIO in Docker\n'
printf '\n  Press %sCtrl-C%s to stop the API; the worker is stopped with it.\n' "$bold" "$reset"
printf '%s==================================================================%s\n\n' "$green" "$reset"

# The API in the FOREGROUND: its logs are this terminal's output and Ctrl-C
# stops it directly, which trips the trap above and takes the worker down too.
# Not `exec` -- exec would replace this shell and the trap would never fire.
"$VENV/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 --reload
