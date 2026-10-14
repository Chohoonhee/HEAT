# Installation

Two environments are used:

| Env | Purpose | Python | Key packages |
|---|---|---|---|
| `heat` | training, open-loop eval, stage-2 processing | 3.8 | torch 1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0, mmseg 0.14.1, bundled mmdetection3d |
| `heat_navsim` | NAVSIM closed-loop eval + navsim cache generation | 3.8 | the `heat` stack **plus** navsim/nuplan dependencies |

Tested on Ubuntu 20.04, CUDA 11.1 driver stack, RTX 3090 / A100.

## 1. Main environment (`heat`)

```bash
conda create -n heat python=3.8 -y
conda activate heat
bash make_env.sh
```

`make_env.sh` performs, in order:

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt --no-deps
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install timm
pip install setuptools==59.5.0        # required for the mmdet3d build
cd mmdetection3d && python setup.py develop && cd ..
pip install nuscenes-devkit==1.1.9
pip install yapf==0.40.1
```

Note: `mmdetection3d/` in this repository is v0.17.1 **with custom Waymo dataset files**
(`mmdet3d/datasets/custom_3d_waymo.py`, `nuscenes_dataset_waymo.py`). Do not replace it
with the upstream package.

## 2. Closed-loop environment (`heat_navsim`)

The closed-loop agent loads the mmdet3d model inside the NAVSIM evaluation loop, so this
environment needs both stacks:

```bash
conda create -n heat_navsim python=3.8 -y
conda activate heat_navsim

# (a) heat stack — same as above (run make_env.sh from the repo root)
bash make_env.sh

# (b) navsim + nuplan dependencies
# nuplan-devkit v1.2 declares py>=3.9; apply the bundled 4-line patch for py3.8
git clone --branch nuplan-devkit-v1.2 https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit && git apply ../navsim_eval/patches/nuplan_py38.patch
pip install -e . --no-deps && cd ..
cd navsim_eval
pip install -e .                                 # navsim + nuplan runtime deps (numpy 1.23.4, hydra 1.2.0, ...)
pip install torchmetrics==0.11.4
pip install pytorch-lightning==1.6.5 --no-deps   # torch-1.9-compatible; --no-deps so pip does not upgrade torch
pip install numba==0.56.4                        # numba compatible with numpy 1.23 (mmdet3d installs 0.48)
pip install setuptools==59.5.0                   # re-pin (navsim install upgrades it; torch 1.9 needs <60)
cd ..
```

## 3. Known quirks

- On Ada/Hopper GPUs (e.g. RTX 4090) the cu111 wheels do not ship matching kernels.
  In `world_model.py`, `waypoint_query_decoder_hist_proto.py`, and
  `waypoint_query_decoder_hist_proto_contrast.py` under `projects/mmdet3d_plugin/LAW/dense_heads/`, replace
  `img2lidars = lidar2img.inverse()` with
  `img2lidars = torch.inverse(lidar2img.cpu()).cuda()`.
- `setuptools==59.5.0` must be installed before `python setup.py develop`
  (newer setuptools breaks the mmdet3d 0.17 build).
