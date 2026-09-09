#!/bin/bash
set -u

REMOTE_HOST=${REMOTE_HOST:-hpd-agai-cust-dev-p4}
REMOTE_REPO=/fsx/sihengx/projects/baozai/IntentionChangeBench
LOCAL_REPO=$(cd "$(dirname "$0")/.." && pwd)
REMOTE_OUTPUT=$REMOTE_REPO/data/simulation/webshop_v2_350_formal.json
REMOTE_COMPLETE=$REMOTE_REPO/data/simulation/webshop_v2_runtime/COMPLETE.json
LOCAL_OUTPUT=$LOCAL_REPO/data/simulation/webshop_v2_350_formal.json
LOCAL_COMPLETE=$LOCAL_REPO/data/simulation/webshop_v2_350_formal.complete.json

while true; do
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE_HOST" \
      "test -s '$REMOTE_OUTPUT' && test -s '$REMOTE_COMPLETE'"; then
    if rsync -av --checksum \
        "$REMOTE_HOST:$REMOTE_OUTPUT" \
        "$LOCAL_OUTPUT" &&
      rsync -av --checksum \
        "$REMOTE_HOST:$REMOTE_COMPLETE" \
        "$LOCAL_COMPLETE" &&
      python3 - "$LOCAL_OUTPUT" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    instances = json.load(handle)
ids = [item.get("instance_id") for item in instances]
if len(instances) != 350 or len(set(ids)) != 350:
    raise SystemExit(f"invalid synchronized output: {len(instances)} cases")
print(f"synchronized and verified {len(instances)} unique cases")
PY
    then
      exit 0
    fi
  fi
  sleep 300
done
