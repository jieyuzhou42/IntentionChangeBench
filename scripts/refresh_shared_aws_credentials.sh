#!/usr/bin/env bash
set -uo pipefail

AUTH_DIR="${1:?auth directory is required}"
STATE_DIR="${2:?state directory is required}"
POLL_SECONDS="${POLL_SECONDS:-300}"

mkdir -p "$AUTH_DIR"
chmod 700 "$AUTH_DIR"
printf '%s\n' \
  '[default]' \
  "credential_process = /bin/cat $AUTH_DIR/credentials.json" \
  'region = us-west-2' \
  > "$AUTH_DIR/config"
chmod 600 "$AUTH_DIR/config"

while [[ ! -f "$STATE_DIR/COMPLETE" && ! -f "$STATE_DIR/FAILED" ]]; do
  temporary="$AUTH_DIR/credentials.json.tmp"
  if aws configure export-credentials --format process > "$temporary"; then
    chmod 600 "$temporary"
    mv "$temporary" "$AUTH_DIR/credentials.json"
  else
    rm -f "$temporary"
  fi
  sleep "$POLL_SECONDS"
done
