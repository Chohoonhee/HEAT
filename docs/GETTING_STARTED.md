# Getting started — HEAT pipeline

All commands run from the repository root in the `heat` environment unless noted otherwise.
Config names passed to the train script are relative to `projects/configs/`
(e.g. `heat/stage3_heat_train` → `projects/configs/heat/stage3_heat_train.py`).

The pipeline has three stages:

| Stage | What it does | Training | Output |
|---|---|---|---|
| 1 | Trajectory-conditioned world-model pretraining | yes | `work_dirs/heat/stage1_world_model/epoch_*.pth` |
| 2 | Trajectory clustering, world-model feature extraction, prototype / action-memory construction | no (inference + averaging) | `proto/18_cluster/all/*.pkl` |
| 3 | HEAT E2E-AD training with the Stage-2 priors | yes | `work_dirs/heat/stage3_heat_train/epoch_*.pth` |

---

## Stage 1 — Trajectory-conditioned world model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/nusc_my_train.sh heat/stage1_world_model 4
```

Trains the latent world model (`World` detector + `world_model` head) jointly on
nuScenes + Waymo + NAVSIM.

## Stage 2 — Trajectory-guided behavior clustering

### 2.1 Trajectory clustering (optional)

The cluster assignments used in the paper are provided in
`clustering_txt/{nus,waymo,nav}_cluster_154k.txt` (one line per training sample, in dataset
order). Each line is `command_index * 10 + cluster_index`, e.g. `24` = command 2, cluster 4.

To regenerate them (K-Means over the first six future waypoints, 6 clusters per driving
command, 18 clusters in total; runs on CPU with images skipped):

```bash
python clustering_txt/cluster_diverse_6_noimg.py --dataset all \
    --cfg projects/configs/heat/stage2_cluster_dataset.py
```

Output: `clustering_txt/generated/all/{nus,waymo,navsim}_cluster_6_154k.txt`.
K-Means is initialization-dependent, so regenerated assignments differ slightly from the
provided ones. To use them, point the `*_cluster_data_dir` entries of the Stage-2/3 configs
at the new files.

### 2.2 Extract world-model features

```bash
CUDA_VISIBLE_DEVICES=0 ./tools/nusc_my_train.sh heat/stage2_extract_feature 1
```

- Loads the Stage-1 checkpoint given by `load_from` in the config.
- One pass over the training sets, writing one pkl per sample (`img_feat`,
  `action_aware_latent`, ...) to `work_dirs/heat/stage2_features/{dataset}/{scene_token}/`.

### 2.3 Build prototypes and action memory

```bash
python proto/noimg_make_proto.py --dataset all                       # visual prototypes P
python proto/noimg_make_proto.py action_aware_latent --dataset all   # visual-coupled action memory M
```

Averages each cluster's latents token by token and writes
`proto/18_cluster/all/{img_feat,action_aware_latent}_prototype_vectors.pkl`, which the
Stage-3 and evaluation configs read. nuScenes provides six cameras and is padded with two
blank views; tokens of the padded views are excluded from these averages.

## Stage 3 — HEAT E2E-AD training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/nusc_my_train.sh heat/stage3_heat_train 8
```

- Model `LAW_proto_contrast` + head `WaypointHead_hist_proto_contrast`
  (Swin-T backbone, 8 camera views, 2 Hz, 4 s horizon), trained with contrastive prototype
  alignment and episodic action-memory refinement.
- 1 sample per GPU (batch size 8 on 8 GPUs), AdamW, learning rate 5e-5 with cosine annealing.
- Output: `work_dirs/heat/stage3_heat_train/epoch_*.pth`.

---

## Open-loop evaluation

```bash
CKPT=work_dirs/heat/stage3_heat_train/latest.pth
CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh projects/configs/heat/openloop_nuscenes.py $CKPT 1
CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh projects/configs/heat/openloop_navsim.py   $CKPT 1
CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh projects/configs/heat/openloop_waymo.py    $CKPT 1
```

The L2 errors at 1 s / 2 s / 3 s are printed as `Average metrics after N calls`.

## Closed-loop evaluation (NAVSIM navtest, PDMS)

Environment `heat_navsim` (see [INSTALL.md](INSTALL.md)); build the metric cache once:

```bash
cd navsim_eval && conda activate heat_navsim
export NAVSIM_EXP_ROOT=exp
./scripts/evaluation/run_metric_caching.sh
cd ..
```

Then:

```bash
./navsim_eval/eval_closedloop_navtest.sh \
    projects/configs/heat/navsim_agent.py $CKPT heat_navtest
```

Per-scenario scores and the aggregate row are written to
`navsim_eval/exp/heat_navtest/<timestamp>/<timestamp>.csv`.

## Troubleshooting

- **`IndexError: list index out of range` in the NAVSIM dataloader**: your
  `data/navsim/cache/train` does not match the sample order of the provided cluster files.
  Use the cache described in [DATA.md](DATA.md), or regenerate the cluster files (Stage 2.1).
- **RTX 4090 / newer GPUs**: see the workaround in [INSTALL.md](INSTALL.md).
