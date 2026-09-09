#!/usr/bin/env bash
set -uo pipefail

REMOTE="${REMOTE:-hpd-agai-cust-pdx}"
REMOTE_REPO="${REMOTE_REPO:-/fsx/sihengx/projects/baozai/IntentionChangeBench}"
REMOTE_RUN="${REMOTE_RUN:-$REMOTE_REPO/data/simulation/travelplanner_formal_run_20260831}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:-$REMOTE_REPO/data/simulation/travelplanner_diverse_360_full_gpt55.json}"
LOCAL_REPO="${LOCAL_REPO:-/home/ANT.AMAZON.COM/sihengx/projects/baozai/IntentionChangeBench}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-$LOCAL_REPO/data/simulation/travelplanner_diverse_360_full_gpt55.json}"
LOCAL_STATUS="${LOCAL_STATUS:-$LOCAL_REPO/data/simulation/travelplanner_formal_status.json}"
LOG="${LOG:-$LOCAL_REPO/data/simulation/travelplanner_formal_sync.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$(dirname "$LOCAL_OUTPUT")"
while true; do
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" \
    "test -f '$REMOTE_RUN/COMPLETE' && test -f '$REMOTE_OUTPUT'"; then
    temporary="${LOCAL_OUTPUT}.syncing"
    if scp -q "$REMOTE:$REMOTE_OUTPUT" "$temporary"; then
      if python3 - "$temporary" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    instances = json.load(handle)
ids = [instance.get("instance_id") for instance in instances]
assert len(instances) == 360, len(instances)
assert len(set(ids)) == 360, "duplicate instance ids"
assert all(len(instance.get("turns", [])) == 7 for instance in instances)
PY
      then
        mv "$temporary" "$LOCAL_OUTPUT"
        scp -q "$REMOTE:$REMOTE_RUN/status.json" \
          "$LOCAL_STATUS" || true
        printf '%s synchronized and validated %s\n' \
          "$(date -u +%FT%TZ)" "$LOCAL_OUTPUT" >> "$LOG"
        exit 0
      fi
      rm -f "$temporary"
    fi
  fi
  printf '%s waiting for remote completion or connectivity\n' \
    "$(date -u +%FT%TZ)" >> "$LOG"
  sleep "$POLL_SECONDS"
done
