#!/bin/bash
# HEAT environment setup (Python 3.8, CUDA 11.1, PyTorch 1.9.1)
# Run from the repository root, inside a fresh conda env:
#   conda create -n heat python=3.8 -y && conda activate heat
#   bash make_env.sh
set -e

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt --no-deps
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install timm==0.9.16   # newer timm needs torch.fx (torch>=1.10)

# mmdetection3d (v0.17.1 sources bundled in this repo, incl. custom Waymo dataset files)
pip install setuptools==59.5.0
cd mmdetection3d
python setup.py develop
cd ..

pip install nuscenes-devkit==1.1.9
pip install yapf==0.40.1
