# Data layout

All configs reference datasets through this directory. Create symlinks to
wherever the actual data lives on your machine:

```
data/
├── nuscenes/                  # nuScenes v1.0 trainval + VAD-format info pkls
│   ├── samples/  sweeps/  maps/  v1.0-trainval/  can_bus/
│   ├── vad_nuscenes_heading_infos_temporal_train.pkl
│   ├── vad_nuscenes_infos_temporal_val.pkl
│   └── nuscenes_map_anns_val.json
├── waymo_e2e_dataset/         # Waymo E2E dataset (processed, 8-view jpg + pkl per frame)
│   ├── waymo_e2e_training_41566/
│   ├── waymo_e2e_val_6000/
│   ├── train_sample_41566_stride2_longseq.txt
│   └── val_sample_heading_6000_stride2.txt
├── navsim/
│   ├── cache/
│   │   ├── train/             # per-sample pkl cache (sample_*.pkl), full navtrain
│   │   └── val/               # per-sample pkl cache, 5,940 samples (open-loop val)
│   └── openscene/             # OpenScene / NAVSIM raw data root
│       ├── navsim_logs/
│       ├── sensor_blobs/
│       └── maps/
```

Example:

```bash
ln -s /path/to/nuscenes                data/nuscenes
ln -s /path/to/waymo_e2e_dataset       data/waymo_e2e_dataset
ln -s /path/to/navsim_cache/train      data/navsim/cache/train
ln -s /path/to/navsim_cache/val        data/navsim/cache/val
ln -s /path/to/openscene               data/navsim/openscene
```

See `docs/DATA.md` for download links and how each part is produced.
