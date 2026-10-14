#!/usr/bin/env bash
# Training entry that also exports NAVSIM environment variables
# (needed when the navsim dataloader requires devkit/map access).
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/my_train_all.sh heat/stage3_heat_train 4
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$REPO_ROOT/data/navsim/openscene/maps"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/navsim_eval/exp}"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT/navsim_eval"
export OPENSCENE_DATA_ROOT="$REPO_ROOT/data/navsim/openscene"

CONFIG=projects/configs/$1.py
GPUS=$2
PORT=${PORT:-27966}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --work-dir work_dirs/$1 \
    --launcher pytorch ${@:3} \
    --deterministic \
    --cfg-options evaluation.jsonfile_prefix=work_dirs/$1/eval/results
