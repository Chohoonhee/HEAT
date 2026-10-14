# Data preparation

HEAT trains jointly on three datasets, all resampled to 2 Hz with a 4-second
(8-waypoint) planning horizon and a unified 8-camera format.

| Dataset | Train samples | Role |
|---|---|---|
| nuScenes | 28,130 | open-loop train/eval (L2, collision) |
| Waymo E2E | 41,566 | open-loop train/eval (L2) |
| NAVSIM (OpenScene) | 85,109 cache pkls | open-loop + closed-loop (PDMS) |

Final on-disk layout: [data/README.md](../data/README.md).

We do not redistribute any dataset. Download each one from its official source under its
own license (§1), then convert it with the scripts in this repository (§2–4).

---

## Unified sample format

Every dataset is converted to the same per-sample structure so a single model can
consume all three.

**Camera order** — 8 views, identical index meaning across datasets:

| idx | Waymo / NAVSIM | nuScenes (VAD) |
|---|---|---|
| 0 | FRONT | CAM_FRONT |
| 1 | FRONT_RIGHT | CAM_FRONT_RIGHT |
| 2 | FRONT_LEFT | CAM_FRONT_LEFT |
| 3 | REAR | CAM_BACK |
| 4 | REAR_LEFT | CAM_BACK_LEFT |
| 5 | REAR_RIGHT | CAM_BACK_RIGHT |
| 6 | SIDE_RIGHT | *(blank)* |
| 7 | SIDE_LEFT | *(blank)* |

nuScenes provides only 6 views, so indices 6–7 are filled with blank images to keep the
input dimension fixed.

**Driving command** (`ego_fut_cmd`, one-hot): `[turn_right, turn_left, go_straight, unknown]`.

**Ego state** (`ego_lcf_feat`): `[vx, vy, ax, ay, vel]`; the model input is
`cat([ego_his_trajs, ego_lcf_feat, ego_fut_cmd])`.

**Trajectories**: `ego_fut_trajs` is `(T, 3)` = `(x, y, heading)` when `with_heading=True`,
sampled at 2 Hz over a 4-second horizon.

---

## 1. Download the official datasets

Register for and download each dataset from its official source, and accept its license:

| Dataset | Source | What to download |
|---|---|---|
| nuScenes | <https://www.nuscenes.org/nuscenes> | v1.0-trainval (all blobs), the map expansion, and the CAN bus expansion |
| Waymo Open Dataset — Vision-based E2E Driving | <https://waymo.com/open/> | the end-to-end driving TFRecords (training and validation) |
| NAVSIM / OpenScene | <https://github.com/autonomousvision/navsim> | `navsim_logs`, `sensor_blobs`, and `maps` (`navsim_eval/download/*.sh` or the [devkit instructions](https://github.com/autonomousvision/navsim/blob/main/docs/install.md)) |

Then build the training data with §2–4 and link everything under `data/` as in
[data/README.md](../data/README.md).

> The NAVSIM cache pkls store image paths that are resolved against
> `data/navsim/openscene/sensor_blobs/...` at load time, so the raw sensor blobs are
> required for NAVSIM **training** as well, not just closed-loop evaluation.

> **Sample lists and cluster files.** The Waymo subset used in the paper is fixed by the
> frame lists in `tools/data_converter/waymo_e2e/splits/`. The provided cluster files
> (`clustering_txt/*_154k.txt`) are row-aligned with the training samples
> (28,130 / 41,566 / 85,109). If your converted data differs in size or ordering, regenerate
> the cluster files (Stage 2.1 in [GETTING_STARTED.md](GETTING_STARTED.md)).

---

## 2. nuScenes

1. Download nuScenes v1.0-trainval + the CAN bus expansion.
2. Generate VAD-format info files **with heading**:

```bash
python tools/data_converter/vad_nuscenes_converter_w_heading.py \
    --root-path data/nuscenes --out-dir data/nuscenes --extra-tag vad_nuscenes_heading
```

Produces `vad_nuscenes_heading_infos_temporal_train.pkl`,
`vad_nuscenes_infos_temporal_val.pkl` and `nuscenes_map_anns_val.json`.
(`vad_nuscenes_converter.py` is the original VAD converter, kept for reference.)

## 3. Waymo E2E

Scripts in `tools/data_converter/waymo_e2e/` (run in that order):

```bash
# a) TFRecord -> per-frame jpg + annotation json  (needs tensorflow + waymo-open-dataset)
#    edit DATASET_FOLDER / TRAIN_FILES / VALIDATION_FILES at the top of the script
python tools/data_converter/waymo_e2e/waymo_extract.py

# b) add ego heading to each frame's annotation json
python tools/data_converter/waymo_e2e/waymo_process_heading.py \
    --root waymo_e2e_training --out waymo_e2e_training_heading

# c) subsample lists: the lists used in the paper are provided in
#    tools/data_converter/waymo_e2e/splits/ -- use them to get the same subset.
#    (To draw a new one-tenth subset instead, edit and run
#     tools/data_converter/waymo_e2e/random_sampling_sequential_stride.py)

# d) copy the listed frames (with images) into the final dataset dir
python tools/data_converter/waymo_e2e/copy_from_txt.py \
    --root waymo_e2e_training_heading \
    --list tools/data_converter/waymo_e2e/splits/train_sample_41566_stride2_longseq.txt \
    --out  waymo_e2e_training_41566 --with-images
python tools/data_converter/waymo_e2e/count_dataset.py       # sanity check
```

Repeat (d) for the validation split with `splits/val_sample_heading_6000_stride2.txt` to
produce `waymo_e2e_val_6000`.

Result: `waymo_e2e_training_41566/`, `waymo_e2e_val_6000/` and the list files
`train_sample_41566_stride2_longseq.txt`, `val_sample_heading_6000_stride2.txt`.
The dataset class picks these list files up automatically via `data_root`
(override with `subsample_file=`).

## 4. NAVSIM cache

The per-sample cache (`sample_*.pkl`) is dumped by iterating the navsim dataloader
(env `heat_navsim`, run from `navsim_eval/`):

```bash
cd navsim_eval
export NUPLAN_MAPS_ROOT=../data/navsim/openscene/maps
export OPENSCENE_DATA_ROOT=../data/navsim/openscene
export NAVSIM_DEVKIT_ROOT=. NAVSIM_EXP_ROOT=exp

# writes pkls under navsim_cache_out/ (paths configured at the top of the script)
python navsim/planning/script/run_training.py agent=law_agent experiment_name=cache_dump

# optional: subsample the dumped cache
python navsim/planning/script/run_train_data_subsampling_gw.py
```

Link the result to `data/navsim/cache/{train,val}`.

## 5. Metric cache for closed-loop evaluation

```bash
cd navsim_eval && conda activate heat_navsim
export NAVSIM_EXP_ROOT=exp
./scripts/evaluation/run_metric_caching.sh   # writes $NAVSIM_EXP_ROOT/metric_cache
```
