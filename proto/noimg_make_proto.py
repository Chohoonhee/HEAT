# projects/mmdet3d_plugin/scripts/debug_vad_dataset.py
import os as _os, sys as _sys
# make `projects.*` importable when run as `python <dir>/<script>.py` from the repo root
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
from mmdet.datasets import build_dataset
from mmdet.datasets import ConcatDataset, RepeatDataset
from mmcv import Config

import torch
import torchvision.transforms as T
from PIL import Image
from PIL import ImageDraw
import numpy as np
import math
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
to_pil = T.ToPILImage()
from pathlib import Path

# normalization에 사용한 mean, std (ImageNet 기준 예시)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def remap_traj_to_vehicle(traj_xyz):
    """
    traj_xyz: (N,3) with current convention ~ (x_right, y_forward, z_up)
    returns (N,3) in Vehicle convention (X_forward, Y_left, Z_up)
    """
    x_right   = traj_xyz[:, 0]
    y_forward = traj_xyz[:, 1]
    z_up      = traj_xyz[:, 2]
    Xv = y_forward + 4.5
    Yv = -x_right
    Zv = z_up + 0.5
    return np.stack([Xv, Yv, Zv], axis=1)

def _as_3x4_projection(P):
    """lidar2img가 3x4 또는 4x4일 수 있어 안전하게 3x4로 변환"""
    P = np.asarray(P)
    if P.shape == (4, 4):
        P = P[:3, :]
    assert P.shape == (3, 4), f"Projection matrix must be 3x4 or 4x4, got {P.shape}"
    return P.astype(np.float64)

# Vehicle/LiDAR → Image 투영
def project_points(points_xyz, projection_matrix):
    """
    points_xyz: (N,3) in LiDAR frame (x forward, y left, z up)
    projection_matrix: lidar2img (3x4 or 4x4)
    return:
      pts_uv: (N,2), depths: (N,) depth in camera coords (homogeneous z)
    """
    P = _as_3x4_projection(projection_matrix)
    N = points_xyz.shape[0]
    pts_h = np.hstack([points_xyz, np.ones((N, 1), dtype=np.float64)])  # (N,4)
    pts_img_h = (P @ pts_h.T).T  # (N,3)
    depths = pts_img_h[:, 2]
    pts_uv = pts_img_h[:, :2] / np.clip(depths[:, None], 1e-6, None)
    return pts_uv, depths

def draw_arrow(draw, p0, p1, color=(0, 255, 0), width=3, head_len=12, head_angle_deg=28):
    """
    draw: PIL.ImageDraw
    p0, p1: (u,v) 시작/끝 픽셀 좌표
    """
    u0, v0 = p0
    u1, v1 = p1
    draw.line([u0, v0, u1, v1], fill=color, width=width)

    dx, dy = (u1 - u0), (v1 - v0)
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    ang = math.radians(head_angle_deg)

    # 역방향 단위벡터에 좌/우 날개 회전
    bx, by = -ux, -uy
    lx = bx * np.cos(ang) - by * np.sin(ang)
    ly = bx * np.sin(ang) + by * np.cos(ang)
    rx = bx * np.cos(-ang) - by * np.sin(-ang)
    ry = bx * np.sin(-ang) + by * np.cos(-ang)

    draw.line([u1, v1, u1 + head_len * lx, v1 + head_len * ly], fill=color, width=width)
    draw.line([u1, v1, u1 + head_len * rx, v1 + head_len * ry], fill=color, width=width)


def safe_load_pkl(pkl_path: Path):
    """파이썬2로 저장된 pkl 섞여 있어도 최대한 복구해서 로드."""
    with open(pkl_path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["nus", "waymo", "nav", "navsim", "all", "nus_waymo", "waymo_navsim", "navsim_nus"],
        required=True,
        help="Which dataset split to debug: nus(NuScenes), waymo(Waymo), nav/navsim(Navsim)."
    )
    parser.add_argument(
        "--cfg",
        # default="projects/configs/law/nus_waymo_navsim_hist.py",  # cluster 만들 때 사용했던 config
        # default='projects/configs/law/nus_waymo_navsim_world_cls_save_wm_latent.py', # 9.4K 버전 config
        default='projects/configs/heat/stage2_make_proto_dataset.py',
        help="Config file path."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="Sample every N examples (default: 100)."
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Drop into pdb after loading each sample."
    )
    return parser.parse_args()

def main(feat_root, cmd_mode=3, cluster_mode=6, ts=8, mode='img_feat'):
   
    # export PYTHONPATH="$(pwd):${PYTHONPATH}"
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
   
    # dataset key & yaw offset 선택
    DATASET_KEYS = {
        "nus":    ("train_nusc",   math.pi / 2),
        "waymo":  ("train_waymo",  math.pi / 2),
        # "nav":    ("train_navsim", math.pi / 2),
        "navsim": ("train_navsim", math.pi / 2),
    }

    targ_datasets = []
    
    if args.dataset =="all":
        datasets = []
        dataset_len=[]
        for key in DATASET_KEYS.keys():
            targ_datasets.append(key)
            ds_key, yaw_offset = DATASET_KEYS[key]
            cur_ds = build_dataset(getattr(cfg.data, ds_key))
            print(f'{key}: {len(cur_ds)}')
            datasets.append(cur_ds)
            dataset_len.append(len(cur_ds))
        ds = ConcatDataset(
            datasets=datasets,
            separate_eval=True  # 훈련은 합치고, 평가 로그는 각자 분리
        )
    
    elif args.dataset == "nus_waymo":
        datasets = []
        dataset_len=[]
        for key in DATASET_KEYS.keys():
            if key == "navsim": continue
            targ_datasets.append(key)
            ds_key, yaw_offset = DATASET_KEYS[key]
            cur_ds = build_dataset(getattr(cfg.data, ds_key))
            print(f'{key}: {len(cur_ds)}')
            datasets.append(cur_ds)
            dataset_len.append(len(cur_ds))
        ds = ConcatDataset(
            datasets=datasets,
            separate_eval=True  # 훈련은 합치고, 평가 로그는 각자 분리
        )

    elif args.dataset == "waymo_navsim":
        datasets = []
        dataset_len=[]
        for key in DATASET_KEYS.keys():
            if key == "nus": continue
            targ_datasets.append(key)
            ds_key, yaw_offset = DATASET_KEYS[key]
            cur_ds = build_dataset(getattr(cfg.data, ds_key))
            print(f'{key}: {len(cur_ds)}')
            datasets.append(cur_ds)
            dataset_len.append(len(cur_ds))
        ds = ConcatDataset(
            datasets=datasets,
            separate_eval=True  # 훈련은 합치고, 평가 로그는 각자 분리
        )
    
    elif args.dataset == "navsim_nus":
        datasets = []
        dataset_len=[]
        for key in DATASET_KEYS.keys():
            if key == "waymo": continue
            targ_datasets.append(key)
            ds_key, yaw_offset = DATASET_KEYS[key]
            cur_ds = build_dataset(getattr(cfg.data, ds_key))
            print(f'{key}: {len(cur_ds)}')
            datasets.append(cur_ds)
            dataset_len.append(len(cur_ds))
        ds = ConcatDataset(
            datasets=datasets,
            separate_eval=True  # 훈련은 합치고, 평가 로그는 각자 분리
        )
        

    else:
        ds_key, yaw_offset = DATASET_KEYS[args.dataset]
        ds = build_dataset(getattr(cfg.data, ds_key))

    # ====== 튜닝 가능한 파라미터 ======
    ARROW_LEN_M = 2.0   # LiDAR 프레임에서 화살 길이(미터)
    DOT_RADIUS  = 8     # 점 반지름(픽셀)
    ARROW_W     = 3     # 선 두께(픽셀)
    # =================================
    navi_trajs = [[], [], [], []]
    traj_idx = [[], [],[], []]
    

    print('total dataset len is: ', len(ds))
    
    clusterwise_proto = None
    clusterwise_count = None
    n_found = 0
    n_missing = 0
    for i in tqdm(range(len(ds)), desc="Processing dataset"):
        # breakpoint()
        # if i > 20 : break
        data = ds[i]
    ### NOTE cluster 생성하는 코드
        ### 대응되는  feat 찾기
        scene_token = data['img_metas'].data[1]['scene_token']
        sample_idx = data['img_metas'].data[1]['sample_idx']
        planning_cluster = data['img_metas'].data[1]['cluster'].data.item()
        dataset_name = data['img_metas'].data[1]["dataset_name"]
        p = data['img_metas'].data[1]['filename'][0]
        image_name = Path(p).stem

        feat_dir = f'{feat_root}/{dataset_name}/{scene_token}'
        feat_path = os.path.join(feat_dir, f'{sample_idx}_{image_name}.pkl')
        # print(feat_path)

        if os.path.exists(feat_path):
            cur_data = safe_load_pkl(feat_path)
            if mode == 'img_feat':
                cur_latent = cur_data['img_feat']
            else:
                cur_latent = cur_data['action_aware_latent']   # img_feat, wm_next_latent, action_aware_latent
            if clusterwise_proto is None:
                # allocate on the first feature actually found (not on i == 0):
                # a missing feature for sample 0 must not leave the accumulators unset
                clusterwise_proto = np.zeros((cmd_mode * cluster_mode,) + cur_latent.shape)
                if mode == 'img_feat':
                    # img_feat is (views, proposals, C): count per view
                    clusterwise_count = np.zeros((cmd_mode * cluster_mode, ts)).astype(np.uint32)
                else:
                    # action latent is (views*proposals, C): count per token row
                    clusterwise_count = np.zeros((cmd_mode * cluster_mode, cur_latent.shape[0])).astype(np.uint32)
            cur_cmd = planning_cluster // 10
            cur_cluster = planning_cluster  % 10
            slot = cur_cmd * cluster_mode + cur_cluster

            n_found += 1
            # nuScenes has no SIDE_RIGHT/SIDE_LEFT: its last two views are blank padding.
            # Exclude those tokens from both the sum and the count (Eqs. (4)-(5)).
            n_view = ts - 2 if dataset_name == 'nuscenes' else ts
            cur_latent = cur_latent.copy()
            if mode == 'img_feat':
                cur_latent[n_view:] = 0
                clusterwise_count[slot, :n_view] += 1
            else:
                n_row = n_view * (cur_latent.shape[0] // ts)
                cur_latent[n_row:] = 0
                clusterwise_count[slot, :n_row] += 1
            clusterwise_proto[slot] += cur_latent

        else:
            n_missing += 1
    if clusterwise_proto is None:
        raise RuntimeError(
            f'No features found under {feat_root}. Run Stage 2.2 (heat/stage2_extract_feature) first.')
    print(f'features aggregated: {n_found} / {len(ds)} samples ({n_missing} missing)')

    # breakpoint()
    print(f'clusterwise_proto shape: {clusterwise_proto.shape}')
    need = clusterwise_proto.ndim - clusterwise_count.ndim
    den = clusterwise_count[(...,) + (None,)* need]
    out = np.zeros_like(clusterwise_proto)
    np.divide(clusterwise_proto, den, out=out, where=den!=0)
    save_pkl = {}
    for i, cur_proto in enumerate(out):
        cur_cmd = i // cluster_mode
        cur_cluster = i % cluster_mode
        cur_planning_cluster = cur_cmd * 10 + cur_cluster
        save_pkl[cur_planning_cluster] = cur_proto
    save_dir = f"proto/{cmd_mode * cluster_mode}_cluster/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    if mode == 'img_feat':
        with open(os.path.join(save_dir, "img_feat_prototype_vectors.pkl"), "wb") as f:
            pickle.dump(save_pkl, f)
    else:
        with open(os.path.join(save_dir, "action_aware_latent_prototype_vectors.pkl"), "wb") as f:
            pickle.dump(save_pkl, f)
    print('Save prototype vector')
    print('End iter sample')    
        
    pass  # (debug breakpoint removed for release)

  
if __name__ == "__main__":
    import sys
    # Usage:
    #   python proto/noimg_make_proto.py                       -> visual prototypes (img_feat)
    #   python proto/noimg_make_proto.py action_aware_latent   -> action memory
    # Features are read from the Stage-2 output directory below.
    ROOT = './work_dirs/heat/stage2_features'
    feat_mode = sys.argv.pop(1) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'img_feat'
    main(ROOT, cmd_mode=3, cluster_mode=6, ts=8, mode=feat_mode)
