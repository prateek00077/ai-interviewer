#!/usr/bin/env bash
#
# Stop what docker-run-script.sh started.
#
#   ./docker-stop-script.sh          Stop the app (API + worker). The datastores
#                                    keep running, so ./docker-run-script.sh or
#                                    ./local-run-script.sh restarts in seconds.
#                                    Nothing is deleted.
#
#   ./docker-stop-script.sh --all    Stop everything: API, worker AND the
#                                    datastores. Your data is kept (the Docker
#                                    volumes survive); the next run brings it
#                                    all back up.
#
#   ./docker-stop-script.sh --wipe   Stop everything and DELETE ALL DATA -- the
#                                    Postgres database and everything in MinIO.
#                                    Asks first. Use it to start completely clean.
#
# (Path B / ./local-run-script.sh runs the app in your terminal, so you stop it
# with Ctrl-C. This script is for the datastores it leaves in Docker: use --all.)

set -euo pipefail
cd "$(dirname "$0")"

bold=$'\033[1m'; green=$'\033[32m'; yellow=$'\033[33m'; red=$'\033[31m'; reset=$'\033[0m'
step()  { printf '\n%s==> %s%s\n' "$bold" "$1" "$reset"; }
info()  { printf '    %s\n' "$1"; }
ok()    { printf '    %s%s%s\n' "$green" "$1" "$reset"; }
warn()  { printf '    %s%s%s\n' "$yellow" "$1" "$reset"; }
die()   { printf '\n%sERROR: %s%s\n' "$red" "$1" "$reset" >&2; exit 1; }

mode="app"
case "${1:-}" in
  "")            mode="app" ;;
  --all)         mode="all" ;;
  --wipe)        mode="wipe" ;;
  -h|--help)
    grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'
    exit 0
    ;;
  *)
    die "Unknown option '${1}'. Use one of: (nothing), --all, --wipe, --help."
    ;;
esac

docker compose version >/dev/null 2>&1 \
  || die "Docker Compose v2 is unavailable ('docker compose'). Is Docker running?"

case "$mode" in
  app)
    step "Stopping the app (API + worker)"
    info "The datastores keep running so the next start is fast. Nothing is deleted."
    docker compose --profile app stop api worker
    ok "API and worker stopped. Datastores (Postgres, Redis, MinIO) are still up."
    info "Bring the app back with ./docker-run-script.sh (or ./local-run-script.sh)."
    ;;

  all)
    step "Stopping everything (app + datastores)"
    info "Your data is kept -- the Docker volumes are not touched."
    docker compose --profile app stop
    ok "All containers stopped. Data preserved."
    info "Bring it all back with ./docker-run-script.sh."
    ;;

  wipe)
    step "Wiping everything (containers + ALL DATA)"
    warn "This DELETES the Postgres database and everything in MinIO. It cannot be undone."
    printf '    Type %swipe%s to confirm: ' "$bold" "$reset"
    read -r answer
    [[ "$answer" == "wipe" ]] || die "Not confirmed (you typed '${answer}'). Nothing was deleted."
    # down removes containers; -v removes the named volumes (the actual data).
    docker compose --profile app down -v
    ok "Everything stopped and all data deleted. This is a clean slate."
    info "./docker-run-script.sh will rebuild the roles and schema from scratch."
    ;;
esac
