#!/usr/bin/env bash
#
# Runs ON THE SERVER. It is piped in over SSH by .github/workflows/deploy.yml:
#
#     ssh root@<ip> "APP_DIR=... DEPLOY_SHA=... bash -s" < deploy/deploy.sh
#
# You can also run it by hand on the server for a manual deploy:
#
#     cd /root/Marketing_tool && bash deploy/deploy.sh
#
# What it does: enter the app directory, pull the exact commit CI verified,
# install any missing dependencies, restart the pm2 process, health-check it —
# and roll back to the previous commit if the new one does not come up healthy.

set -euo pipefail

# Defaults match the live server: /root/Marketing_tool, pm2 process
# "marketing-tool", virtualenv in venv/. Override any of them with repo variables.
APP_DIR="${APP_DIR:-/root/Marketing_tool}"
PM2_NAME="${PM2_NAME:-marketing-tool}"
BRANCH="${BRANCH:-main}"
DEPLOY_SHA="${DEPLOY_SHA:-}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:3001/health}"
HEALTH_RETRIES="${HEALTH_RETRIES:-15}"   # x 2s = up to 30s for the app to boot
PYTHON="${PYTHON:-python3}"

log() { printf '\n==> %s\n' "$*"; }

# --- preflight --------------------------------------------------------------
if [ ! -d "$APP_DIR" ]; then
  echo "ERROR: APP_DIR '$APP_DIR' does not exist on $(hostname)." >&2
  echo "       Set the APP_DIR repository variable to the real path." >&2
  exit 1
fi
cd "$APP_DIR"

if [ ! -d .git ]; then
  echo "ERROR: $APP_DIR is not a git checkout. See deploy/README.md for first-time setup." >&2
  exit 1
fi

# The venv is venv/ on this server; .venv/ is what a fresh setup creates. Use
# whichever is already there so a deploy never builds a second one alongside it.
if   [ -x venv/bin/python ];  then VENV="venv"
elif [ -x .venv/bin/python ]; then VENV=".venv"
else VENV="${VENV:-venv}"
fi

# Not fatal: config may come from the pm2 ecosystem file's env block instead of a
# .env file. If it is genuinely missing, the health check below catches it.
if [ ! -f .env ]; then
  echo "WARNING: $APP_DIR/.env not found — assuming pm2 supplies the environment." >&2
fi

# pm2 is per-user: root's pm2 and deploy's pm2 are separate daemons with separate
# process lists. This script talks to whichever belongs to the user running it.
#
# A non-interactive `ssh host "cmd"` does not source ~/.bashrc, so an nvm-installed
# pm2 is missing from PATH here even though it works fine when you log in by hand.
# That is the usual cause of "pm2: command not found" in a deploy that works
# manually — so load nvm ourselves before giving up.
if ! command -v pm2 >/dev/null 2>&1; then
  for nvm_sh in "$HOME/.nvm/nvm.sh" /usr/local/nvm/nvm.sh; do
    if [ -s "$nvm_sh" ]; then
      # shellcheck disable=SC1090
      . "$nvm_sh" >/dev/null 2>&1 || true
      break
    fi
  done
fi

if ! command -v pm2 >/dev/null 2>&1; then
  echo "ERROR: pm2 is not on PATH for $(whoami)." >&2
  echo "       Install it with 'npm install -g pm2', or find it with" >&2
  echo "       'command -v pm2' when logged in and add that directory to PATH." >&2
  exit 1
fi

log "Using pm2 at $(command -v pm2) as $(whoami)"

# git refuses to touch a repo owned by another user ("dubious ownership"), which
# is exactly the case when root deploys a checkout owned by the service user.
# (--add is guarded, or every deploy would append another duplicate line.)
if ! git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$APP_DIR"; then
  git config --global --add safe.directory "$APP_DIR" 2>/dev/null || true
fi

# Whoever owns the checkout is who the app runs as; deploying as root would
# otherwise leave root-owned files behind for it to choke on.
OWNER="$(stat -c '%U:%G' .)"

PREVIOUS_SHA="$(git rev-parse HEAD)"
log "Current commit: $(git log -1 --oneline)"

log "Fetching origin"
git fetch --prune --quiet origin

# The workflow passes the exact commit it verified. Falling back to the branch tip
# only matters for manual runs.
if [ -z "$DEPLOY_SHA" ]; then
  DEPLOY_SHA="$(git rev-parse "origin/$BRANCH")"
fi

if [ "$DEPLOY_SHA" = "$PREVIOUS_SHA" ]; then
  log "Already at $DEPLOY_SHA — reinstalling and restarting anyway"
fi

# --- helpers ----------------------------------------------------------------

# Check out a commit, install missing dependencies, restart the pm2 process.
release() {
  local sha="$1"

  log "Checking out $sha"
  git checkout --quiet "$BRANCH" 2>/dev/null || git checkout --quiet -B "$BRANCH" "origin/$BRANCH"
  git reset --hard --quiet "$sha"

  if [ ! -x "$VENV/bin/python" ]; then
    log "Creating virtualenv at $VENV"
    "$PYTHON" -m venv "$VENV"
  fi

  # pip only downloads what is missing or version-mismatched, so this is cheap on
  # a deploy that did not touch requirements.txt.
  log "Installing dependencies into $VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -r requirements.txt

  if [ "$(id -u)" -eq 0 ] && [ "$OWNER" != "root:root" ]; then
    log "Restoring ownership to $OWNER"
    chown -R "$OWNER" "$APP_DIR"
  fi

  # 'pm2 restart' fails if the process was never registered (first deploy, or the
  # pm2 daemon was reset), so fall back to ecosystem.config.js. It is TRACKED, so
  # 'git reset --hard' above has just set it to this commit's version - an edit
  # made to it on the server does not survive a deploy. It declares both the API
  # and the Vision Lab worker, so this path starts both.
  if pm2 describe "$PM2_NAME" >/dev/null 2>&1; then
    log "Restarting pm2 process $PM2_NAME"
    pm2 restart "$PM2_NAME" --update-env
  else
    local cfg=""
    for candidate in ecosystem.config.js ecosystem.config.json deploy/pm2.config.json; do
      [ -f "$candidate" ] && { cfg="$candidate"; break; }
    done
    if [ -z "$cfg" ]; then
      echo "ERROR: pm2 has no process named '$PM2_NAME' and no config file to start it from." >&2
      echo "       'pm2 list' as $(whoami) shows what is registered — set the PM2_NAME" >&2
      echo "       repository variable to one of those names." >&2
      exit 1
    fi
    log "pm2 process $PM2_NAME not found — starting it from $cfg"
    pm2 start "$cfg"
  fi

  # Persist the process list so it survives a reboot (needs 'pm2 startup' once).
  pm2 save --force >/dev/null 2>&1 || true
}

# Restart the Vision Lab worker - a second pm2 process - onto the new code.
#
# Without this the worker keeps running the PREVIOUS release: the API moves on,
# the worker does not, and a fix that lives in the worker ships and changes
# nothing. It is found by WHAT IT RUNS (vision_lab.worker), not by name, so a
# worker started by hand under any name is restarted too.
#
# Deliberately never fatal and never a start: a missing worker is a warning, and
# this does not launch one - a second worker the operator did not ask for would
# double the CPU load on a small box.
restart_workers() {
  local ids
  ids="$(pm2 jlist 2>/dev/null | "$VENV/bin/python" -c '
import json, sys
try:
    procs = json.load(sys.stdin)
except Exception:
    procs = []
found = []
for p in procs:
    env = p.get("pm2_env") or {}
    args = env.get("args") or []
    if isinstance(args, str):
        args = [args]
    runs = " ".join([str(env.get("pm_exec_path") or "")] + [str(a) for a in args])
    if "vision_lab.worker" in runs or "vision_lab/worker" in runs:
        found.append(str(p.get("pm_id")))
print(" ".join(found))
' 2>/dev/null || true)"

  if [ -z "$ids" ]; then
    log "No Vision Lab worker is registered in pm2 - nothing to restart"
    echo "WARNING: Vision Lab analyses will stay queued until a worker runs. Start it once with:" >&2
    echo "         pm2 start ecosystem.config.js --only vision-worker && pm2 save" >&2
    return 0
  fi
  for id in $ids; do
    log "Restarting Vision Lab worker (pm2 id $id)"
    pm2 restart "$id" --update-env || echo "WARNING: could not restart worker $id" >&2
  done
  pm2 save --force >/dev/null 2>&1 || true
}

# Poll /health until it answers or we run out of retries.
healthy() {
  local i
  for i in $(seq 1 "$HEALTH_RETRIES"); do
    if curl --fail --silent --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

# --- deploy -----------------------------------------------------------------
release "$DEPLOY_SHA"

if healthy; then
  log "Healthy. Deployed: $(git log -1 --oneline)"
  # Only now, with the API confirmed healthy: a rollback below never has to
  # undo a worker restart, so API and worker can never end up on different
  # releases because of a failed deploy.
  restart_workers
  curl --silent --max-time 5 "$HEALTH_URL" || true
  echo
  pm2 list
  exit 0
fi

# --- rollback ---------------------------------------------------------------
log "HEALTH CHECK FAILED at $HEALTH_URL — rolling back to $PREVIOUS_SHA"
pm2 list || true
pm2 logs "$PM2_NAME" --lines 60 --nostream || true

release "$PREVIOUS_SHA"

if healthy; then
  log "Rolled back to $(git log -1 --oneline) and it is healthy. The new commit was NOT deployed."
else
  log "ROLLBACK IS ALSO UNHEALTHY — the service is down and needs manual attention."
fi
exit 1
