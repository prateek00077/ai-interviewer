#!/usr/bin/env bash
#
# One-command local setup for AI Interviewer (all-in-Docker, README Path A).
#
# Run it, then open http://localhost:8000/dev in a browser to test:
#
#   ./setup.sh
#
# The only thing you must do by hand is put a real NVIDIA API key in .env
# (get one at https://build.nvidia.com -- it starts `nvapi-`). This script
# refuses to go further without it.
#
# It is safe to re-run: every step is idempotent, so `./setup.sh` also serves
# as "bring the stack back up". To wipe everything and start clean:
#
#   docker compose --profile app down && docker compose down -v
#
# ...then run ./setup.sh again (it re-creates the DB roles that the volumes hold).

set -euo pipefail

# Always operate from the repo root (this script's own directory).
cd "$(dirname "$0")"

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
  || die "Docker is not installed or not on PATH. Install Docker Desktop / Engine and retry."

docker info >/dev/null 2>&1 \
  || die "The Docker daemon is not reachable. Start Docker and retry."

docker compose version >/dev/null 2>&1 \
  || die "Docker Compose v2 is unavailable ('docker compose'). Update Docker and retry."

ok "Docker and Docker Compose are available."

if [[ ! -f .env ]]; then
  cp .env.example .env
  die ".env did not exist, so I created it from .env.example.
    Now open .env and set NVIDIA_API_KEY to your key (get one at https://build.nvidia.com,
    it starts 'nvapi-'), then re-run ./setup.sh."
fi

# The key must be present and not the shipped placeholder.
key_line="$(grep -E '^[[:space:]]*NVIDIA_API_KEY=' .env | tail -n1 || true)"
key_val="${key_line#*=}"
if [[ -z "$key_val" || "$key_val" == nvapi-xxxx* || "$key_val" != nvapi-* ]]; then
  die "NVIDIA_API_KEY in .env is missing or still the placeholder.
    Get a key at https://build.nvidia.com (it starts 'nvapi-'), set this line in .env:
        NVIDIA_API_KEY=nvapi-your-key-here
    then re-run ./setup.sh. There is no offline fallback -- the interview needs it."
fi
ok "NVIDIA_API_KEY is set."

# --- 2. datastores up -------------------------------------------------------
step "Starting datastores (Postgres, Redis, MinIO)"
docker compose up -d

info "Waiting for Postgres, Redis and MinIO to report healthy..."
deadline=$(( SECONDS + 120 ))
while :; do
  # Every declared healthcheck must be 'healthy'. `docker compose ps` with the
  # health filter lists only containers still lacking a healthy status.
  unhealthy="$(docker compose ps --format '{{.Service}} {{.Health}}' \
    | awk '$2 != "" && $2 != "healthy" { print $1 }' || true)"
  if [[ -z "$unhealthy" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    docker compose ps
    die "Timed out waiting for datastores to become healthy: ${unhealthy//$'\n'/, }"
  fi
  sleep 3
done
ok "Datastores are healthy."

# --- 3. bootstrap DB roles (idempotent) ------------------------------------
step "Bootstrapping database roles and pgvector"
# Guarded by IF NOT EXISTS / ALTER inside the SQL, so this is safe every run.
# Passwords match the URLs shipped in .env.example.
docker compose exec -T postgres psql -U postgres -d ai_interviewer \
  -v owner_pw="'owner_pw'" -v app_pw="'app_pw'" < scripts/bootstrap_db.sql
ok "Roles (app_owner, app_user, app_auth) and the vector extension are in place."

# --- 4. build & start API + worker -----------------------------------------
step "Building and starting the API and worker"
warn "First build takes ~15-20 min and produces a ~1.7 GB image (torch + opencv)."
warn "Later runs are cached and finish in seconds -- this wait is not a hang."
docker compose --profile app up -d --build
ok "API (aii-api) and worker (aii-worker) are up."

# --- 5. migrate schema ------------------------------------------------------
step "Applying database migrations"
docker compose exec -T api alembic upgrade head
ok "Schema is at head (expected 0012)."

# --- 6. seed demo tenant ----------------------------------------------------
step "Seeding a demo tenant"
info "Login credentials are printed below by seed_data.py -- keep them handy:"
docker compose exec -T api python scripts/seed_data.py
ok "Demo org, recruiter and candidate created."

# --- 7. verify it came up ---------------------------------------------------
step "Verifying the app is ready"
deadline=$(( SECONDS + 90 ))
ready_body=""
while :; do
  if ready_body="$(curl -fsS http://localhost:8000/ready 2>/dev/null)"; then
    break
  fi
  if (( SECONDS >= deadline )); then
    # One last call without -f so we can show the 503 body naming the culprit.
    ready_body="$(curl -sS http://localhost:8000/ready 2>/dev/null || true)"
    die "/ready did not come up. Last response: ${ready_body:-<none>}
    Check logs with:  docker compose logs -f api worker"
  fi
  sleep 2
done
ok "/ready: $ready_body"

# NVIDIA endpoints are a warning, not a blocker: the stack is up regardless and
# a bad key can be fixed in .env + `docker compose --profile app restart api worker`.
if docker compose exec -T api python scripts/check_nim.py >/dev/null 2>&1; then
  ok "NVIDIA NIM endpoints (LLM, ASR, TTS) all responded."
else
  warn "NVIDIA NIM check failed -- interviews will be silent until it passes."
  warn "Run 'docker compose exec api python scripts/check_nim.py' to see which service and why (usually a bad or rate-limited key)."
fi

# --- 8. final banner --------------------------------------------------------
printf '\n%s==================================================================%s\n' "$green" "$reset"
printf '%s  AI Interviewer is up. Open this in your browser to test:%s\n' "$bold" "$reset"
printf '\n      %s%shttp://localhost:8000/dev%s   <- the test console\n' "$bold" "$green" "$reset"
printf '\n  Other URLs:\n'
printf '      API docs       http://localhost:8000/docs\n'
printf '      MinIO console  http://localhost:9001  (minioadmin / minioadmin)\n'
printf '\n  While testing:\n'
printf '      - After uploading a resume, wait for the panel to reach READY\n'
printf '        before generating the plan (a too-early plan ignores the resume).\n'
printf '      - Plan generation takes 30-60s. That is normal.\n'
printf '      - Follow logs:  docker compose logs -f api worker\n'
printf '%s==================================================================%s\n\n' "$green" "$reset"
