#!/usr/bin/env bash
# HEAT closed-loop evaluation (PDM score) on the NAVSIM navtest split.
#
# Usage:
#   cd navsim_eval
#   ./eval_closedloop_navtest.sh [CONFIG] [CHECKPOINT] [EXPERIMENT_NAME]
#
# CHECKPOINT is required (a Stage-3 checkpoint, e.g. work_dirs/heat/stage3_heat_train/latest.pth).
# Requires (see docs/DATA.md):
#   - data/navsim/openscene/{navsim_logs,sensor_blobs,maps}
#   - metric cache at $NAVSIM_EXP_ROOT/metric_cache
#     (generate once with: ./scripts/evaluation/run_metric_caching.sh)
set -e

NAVSIM_ROOT="$(cd "$(dirname "$0")" && pwd)"
HEAT_ROOT="$(cd "$NAVSIM_ROOT/.." && pwd)"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$HEAT_ROOT/data/navsim/openscene/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$HEAT_ROOT/data/navsim/openscene}"

CONFIG=${1:-$HEAT_ROOT/projects/configs/heat/navsim_agent.py}
CKPT=${2:?usage: eval_closedloop_navtest.sh CONFIG CHECKPOINT [EXPERIMENT_NAME]}
EXP=${3:-heat_closedloop_navtest}

# run from the repository root so the agent config's relative paths
# (./proto/*.pkl, ./clustering_txt/*.txt, data/...) resolve correctly
cd "$HEAT_ROOT"

SINGLE_WORKER=1 python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_vis_250917.py" \
    train_test_split=navtest agent=law_agent worker=sequential \
    experiment_name="$EXP" \
    agent.law_model_config="$CONFIG" \
    agent.law_model_checkpoint="$CKPT"
