# HEAT: Heterogeneous End-to-End Autonomous Driving via Trajectory-Guided World Models

**IEEE Robotics and Automation Letters (RA-L), 2026**

[Hoonhee Cho](https://chohoonhee.github.io/hoonheecho/)\*, [Giwon Lee](https://giwonlee00.github.io/)\*, [Jae-Young Kang](https://mickeykang16.github.io/)\*, [Hyemin Yang](https://hemin0806.github.io/)\*, [Heejun Park](https://parkie0517.github.io/)\*, [Kuk-Jin Yoon](https://vi.kaist.ac.kr/) (\*equal contribution) — KAIST

[**Project Page**](https://heat2026.github.io/) | [**arXiv**](https://arxiv.org/abs/2605.19631)

Official implementation of HEAT. A single end-to-end driving model is trained jointly on
nuScenes, NAVSIM, and Waymo by organizing learning around planning trajectories.

## Overview

1. **Stage 1**: pretrain a trajectory-conditioned latent world model.
2. **Stage 2**: cluster samples by their planning trajectories (18 clusters) and build visual
   prototypes and an action memory from the world model (no training).
3. **Stage 3**: train the planner with prototype contrastive learning and memory-based action refinement.

## Results

Single unified model trained jointly on nuScenes + NAVSIM + Waymo (numbers reported in the paper):

| Benchmark | Metric | HEAT |
|---|---|---|
| nuScenes | Avg. L2 (m) / Avg. Col. (%) | 0.35 / 0.17 |
| NAVSIM (open-loop) | Avg. L2 (m) | 0.22 |
| Waymo E2E (sampled subset) | Avg. L2 (m) | 0.68 |
| NAVSIM navtest (closed-loop) | PDMS | 85.2 |

See the paper for the full tables.

## Installation

```bash
conda create -n heat python=3.8 -y
conda activate heat
bash make_env.sh          # torch 1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0, bundled mmdetection3d
```

Details (including the closed-loop evaluation environment):
[docs/INSTALL.md](docs/INSTALL.md)

## Data

Set up `data/` as described in [data/README.md](data/README.md); download links and
from-scratch preprocessing instructions are in [docs/DATA.md](docs/DATA.md).

## Training & evaluation

The full pipeline, with the inputs and outputs of every stage, is documented in
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). In short:

```bash
# Stage 1: trajectory-conditioned world model (all three datasets)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/nusc_my_train.sh heat/stage1_world_model 4

# Stage 2: world-model features -> prototypes / action memory
#   (trajectory cluster assignments are provided in clustering_txt/;
#    regenerate them with clustering_txt/cluster_diverse_6_noimg.py if needed)
CUDA_VISIBLE_DEVICES=0 ./tools/nusc_my_train.sh heat/stage2_extract_feature 1
python proto/noimg_make_proto.py --dataset all                       # visual prototypes
python proto/noimg_make_proto.py action_aware_latent --dataset all   # action memory

# Stage 3: HEAT E2E-AD training
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/nusc_my_train.sh heat/stage3_heat_train 4

# Open-loop evaluation (nuScenes / NAVSIM / Waymo)
CKPT=work_dirs/heat/stage3_heat_train/latest.pth
CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh projects/configs/heat/openloop_nuscenes.py $CKPT 1
CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh projects/configs/heat/openloop_navsim.py   $CKPT 1
CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh projects/configs/heat/openloop_waymo.py    $CKPT 1

# Closed-loop evaluation (NAVSIM navtest, PDM score)
./navsim_eval/eval_closedloop_navtest.sh projects/configs/heat/navsim_agent.py $CKPT heat_navtest
```

## Repository structure

```
projects/            mmdet3d plugin (models, heads, datasets) + configs (projects/configs/heat)
mmdetection3d/       bundled mmdetection3d v0.17.1 (with custom Waymo dataset support)
tools/               train/test entry points and data converters
clustering_txt/      trajectory clustering script + cluster assignments used in the paper
proto/               prototype / action-memory build script
navsim_eval/         NAVSIM devkit fork for closed-loop PDM-score evaluation
data/                dataset root (see data/README.md)
```

## Acknowledgements

This codebase builds on
[LAW](https://github.com/BraveGroup/LAW),
[VAD](https://github.com/hustvl/VAD),
[mmdetection3d](https://github.com/open-mmlab/mmdetection3d),
[NAVSIM](https://github.com/autonomousvision/navsim), and
[DiffusionDrive](https://github.com/hustvl/DiffusionDrive).
We thank the authors for their great work.

## Citation

```bibtex
@article{cho2026heat,
  title={HEAT: Heterogeneous End-to-End Autonomous Driving via Trajectory-Guided World Models},
  author={Cho, Hoonhee and Lee, Giwon and Kang, Jae-Young and Yang, Hyemin and Park, Heejun and Yoon, Kuk-Jin},
  journal={arXiv preprint arXiv:2605.19631},
  year={2026}
}
```
