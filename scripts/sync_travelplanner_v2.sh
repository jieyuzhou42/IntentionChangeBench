#!/usr/bin/env bash
set -uo pipefail
REMOTE="${REMOTE:-hpd-agai-cust-pdx}"
REMOTE_RUN="${REMOTE_RUN:?REMOTE_RUN is required}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:?REMOTE_OUTPUT is required}"
LOCAL_REPO="${LOCAL_REPO:-/home/ANT.AMAZON.COM/sihengx/projects/baozai/IntentionChangeBench}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-$LOCAL_REPO/data/simulation/travelPlanner_v2.json}"
LOCAL_STATUS="${LOCAL_STATUS:-$LOCAL_REPO/data/simulation/travelPlanner_v2.status.json}"
LOG="${LOG:-$LOCAL_REPO/data/simulation/travelPlanner_v2.sync.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
mkdir -p "$(dirname "$LOCAL_OUTPUT")"
while true; do
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "test -f '$REMOTE_RUN/FAILED'"; then
    scp -q "$REMOTE:$REMOTE_RUN/status.json" "$LOCAL_STATUS" || true
    printf '%s remote run failed\n' "$(date -u +%FT%TZ)" >> "$LOG"
    exit 2
  fi
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "test -f '$REMOTE_RUN/COMPLETE' && test -f '$REMOTE_OUTPUT'"; then
    temporary="${LOCAL_OUTPUT}.syncing"
    if scp -q "$REMOTE:$REMOTE_OUTPUT" "$temporary" \
      && python3 "$LOCAL_REPO/scripts/validate_travelplanner_v2.py" \
        --tasks "$LOCAL_REPO/data/travelplanner/short_horizon_250_multi_360_tasks.json" \
        --output "$temporary"; then
      mv "$temporary" "$LOCAL_OUTPUT"
      scp -q "$REMOTE:$REMOTE_RUN/status.json" "$LOCAL_STATUS" || true
      printf '%s synchronized and validated %s\n' "$(date -u +%FT%TZ)" "$LOCAL_OUTPUT" >> "$LOG"
      exit 0
    fi
    rm -f "$temporary"
  fi
  printf '%s waiting for remote completion or connectivity\n' "$(date -u +%FT%TZ)" >> "$LOG"
  sleep "$POLL_SECONDS"
done
