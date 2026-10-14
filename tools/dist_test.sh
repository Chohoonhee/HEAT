#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
# PORT=${PORT:-31166}
if [ -z "${PORT}" ]; then
  while :; do
    CANDIDATE=$(shuf -i 20000-65000 -n 1)
    if ! lsof -iTCP:"$CANDIDATE" -sTCP:LISTEN -Pn >/dev/null 2>&1; then
      PORT="$CANDIDATE"
      break
    fi
  done
fi
echo "Using PORT=$PORT"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
