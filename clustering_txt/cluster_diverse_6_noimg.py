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
to_pil = T.ToPILImage()

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
        default='projects/configs/heat/stage2_cluster_dataset.py',
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

def main():
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
    
    for i in tqdm(range(len(ds)), desc="Processing dataset"):
        # breakpoint()
        # if i > 10 : break
        data = ds[i]
    ### NOTE cluster 생성하는 코드
        plan_traj = data['ego_fut_trajs'].data[0, :6, :2].cumsum(0).numpy().astype(np.int32)
        plan_mask = data['ego_fut_masks'].data[0][0][:6].numpy()
        cmd = data['ego_fut_cmd'].data[0][0].numpy().astype(np.int32)
        cmd = cmd.argmax()
        navi_trajs[cmd].append(plan_traj)
        traj_idx[cmd].append(i)
    
    Ks = [6]
    # K=6
    clusters = [[], [], [], []]
    cluster_per_idx = np.zeros((4, len(ds)))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    # labels_per_cluster = []
    # breakpoint()
    for idx, trajs in enumerate(navi_trajs):
        # breakpoint()
        for i,K in enumerate(Ks):
            if trajs == []:
                # 없는 경우: 더미 값으로 채움 (예: 0)
                dummy = np.zeros((K, 6, 2))  # or np.nan if you want to distinguish
                clusters.append(dummy)
                # breakpoint()
                continue
            trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
            kmeans = KMeans(n_clusters=K).fit(trajs)
            cluster = kmeans.cluster_centers_
            labels = kmeans.labels_
            cluster = cluster.reshape(-1, 6, 2)
            clusters[i].append(cluster)
            # labels_per_cluster.append(labels)
            cluster_per_idx[i][traj_idx[idx]] = idx * 10 + labels
            for j in range(K):
                axes[i].scatter(cluster[j, :, 0], cluster[j, :,1])
                axes[i].set_title(f'Cluster{K}')
    # breakpoint()
    save_dir = f'clustering_txt/generated/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/plan_total_154K', bbox_inches='tight')
    plt.close()

    for j,K in enumerate(Ks):
        cur_clusters = np.stack(clusters[j], axis=0)
        np.save(f'{save_dir}/kmeans_plan_{K}_total_154K.npy', cur_clusters)
        cur_cluster_per_idx = cluster_per_idx[j]
        offset = 0
        # dataset_names = [ds for ds in DATASET_KEYS.keys()]
        for i, dlen in enumerate(dataset_len):
            splitted_cluster_per_idx = cur_cluster_per_idx[offset: offset+dlen]
            save_path = f'{save_dir}/{targ_datasets[i]}_cluster_{K}_154k.txt'
            np.savetxt(save_path, splitted_cluster_per_idx, fmt="%d")
            offset = offset+dlen
                
    pass  # (debug breakpoint removed for release)

  
if __name__ == "__main__":
    main()
