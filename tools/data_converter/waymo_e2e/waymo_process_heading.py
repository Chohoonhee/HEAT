#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

def inverse_T(T):
    assert T.shape == (4, 4)
    R = T[:3, :3]
    R_inv = np.linalg.inv(R)
    t = T[:-1, -1]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:-1, -1] = -R_inv @ t
    return T_inv

EPS = 1e-3

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rot_from_T(T):
    """4x4 extrinsic → 3x3 회전행렬"""
    R = np.array(T, dtype=np.float64).reshape(4,4)[:3,:3]
    # 직교화 (수치오차 보정)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R

def omega_cam_to_vehicle(omega_cam, T_c2v):
    """
    카메라 기준 각속도 → Vehicle 기준 각속도
    omega_cam: dict {'wX','wY','wZ'} 또는 array-like
    T_c2v: camera->vehicle extrinsic (4x4)
    """
    R_c2v = rot_from_T(T_c2v)
    if isinstance(omega_cam, dict):
        w_c = np.array([omega_cam['wX'], omega_cam['wY'], omega_cam['wZ']], dtype=np.float64)
    else:
        w_c = np.array(omega_cam, dtype=np.float64).reshape(3)
    return R_c2v @ w_c

def get_v2c_from_cam2veh(extrinsic_list):
    """
    Waymo extrinsic: T_cam2veh (flattened 4x4) -> T_v2c (4x4 homogeneous)
    X_cam = R_v2c @ X_veh + t_v2c
    """
    T_cam2veh = np.array(extrinsic_list, dtype=float).reshape(4, 4)

    

    R_c2v = T_cam2veh[:3, :3]
    t_c2v = T_cam2veh[:3, 3]

    R_v2c = R_c2v.T
    t_v2c = -R_v2c @ t_c2v

    T_v2c = np.eye(4, dtype=float)
    T_v2c[:3, :3] = R_v2c
    T_v2c[:3, 3] = t_v2c

    axes_tf = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]])
    new_T_v2c = np.matmul(axes_tf, T_v2c)
    return new_T_v2c

def yaw_from_xy_forward(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = np.zeros_like(x); dy = np.zeros_like(y)
    dx[:-1] = x[1:] - x[:-1]
    dy[:-1] = y[1:] - y[:-1]
    # 마지막 스텝은 직전 방향 재사용 (정지 등 방지)
    if len(x) >= 2 and (dx[-2] != 0 or dy[-2] != 0):
        dx[-1] = dx[-2]; dy[-1] = dy[-2]
    yaw = np.arctan2(dy, dx)      # [-pi, pi]
    yaw = np.unwrap(yaw)          # 연속화
    return yaw

def extract_sequence_info(json_list):
    """
    시퀀스 전체 JSON 데이터를 보고 공통 정보를 추출.
    TODO: 네가 원하는 로직 작성.
    """
    
    omega = []
    for i, js in enumerate(json_list):
        pass  # debug breakpoint removed for release
        front_omega_c = js['metadata']['velocity']['front']      # dict with wX,wY,wZ
        # sideL_omega_c = js['metadata']['velocity']['side_left']  # dict with wX,wY,wZ

        T_c2v_front = np.array(js['metadata']['camera_calibrations']['front']['extrinsic']).reshape(4,4)
        # T_c2v_sideL = np.array(js['metadata']['camera_calibrations']['side_left']['extrinsic']).reshape(4,4)

        w_front_in_vehicle = omega_cam_to_vehicle(front_omega_c, T_c2v_front)
        omega.append(w_front_in_vehicle)
        # w_sideL_in_vehicle = omega_cam_to_vehicle(sideL_omega_c, T_c2v_sideL)
        
        # T_v2c = get_v2c_from_cam2veh(js['metadata']['camera_calibrations']['front']['extrinsic'])
        # T_c2v = inverse_T(T_v2c)
    omega = np.stack(omega)
    return omega
    

def main(root, out, pattern):
    root = Path(root)
    out = Path(out)

    for seq_dir in root.iterdir():
        if not seq_dir.is_dir():
            continue

        # json_files = list(seq_dir.glob(pattern))
        json_files = sorted(seq_dir.glob(pattern), key=lambda p: p.name)

        if not json_files:
            continue

        # 1) 모든 JSON 로드
        all_data = []
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                all_data.append(json.load(f))

        # 2) 시퀀스 단위 정보 추출
        # seq_info = extract_sequence_info(all_data)
        # seq_mask = np.ones(len(seq_info), dtype=int)
        
        # pad_ = 20
        # seq_info = np.concatenate([np.zeros([pad_, 3]), seq_info, np.zeros([pad_, 3])], axis=0)
        # seq_mask = np.concatenate([np.zeros(pad_), seq_mask, np.zeros(pad_)], axis=0)
        
        
        # 3) 각 JSON에 추가 필드 붙여 저장
        for idx, (jf, data) in enumerate(zip(json_files, all_data)):
            # 최상위에 직접 붙여도 되고, _meta 밑에 넣어도 됨
            # omega_idx = pad_ + i

            ps = data['metadata']['past_states']
            fs = data['metadata']['future_states']

            # ---------- PAST: 인덱스 -n ... -1, 0 (길이 Np) ----------
            xp = np.asarray(ps['pos_x'], float)
            yp = np.asarray(ps['pos_y'], float)

            dx_p = np.diff(xp)
            dy_p = np.diff(yp)
            yaw_seg_p = np.arctan2(dy_p, dx_p)   # 세그먼트 방향, 길이 Np-1
            spd_p = np.hypot(dx_p, dy_p)

            Np = len(xp)
            yaw_past = np.empty(Np, float)
            yaw_past[-1] = 0.0                   # **현재(0) 시점은 정의상 0**

            # 현재(-1) → 과거(…0)로 역전파
            for i in range(Np - 2, -1, -1):
                if spd_p[i] < EPS:
                    yaw_past[i] = yaw_past[i+1]
                else:
                    prev = yaw_past[i+1]
                    # 세그먼트 방향을 prev 근처로 연속화해서 붙임
                    yaw_past[i] = prev + wrap_to_pi(yaw_seg_p[i] - prev)

            # ---------- FUTURE: 인덱스 +1, +2, ... (길이 Nf) ----------
            xf = np.asarray(fs['pos_x'], float)
            yf = np.asarray(fs['pos_y'], float)

            # 현재(0,0) → +1, +1→+2, ...
            dx_f = np.diff(np.r_[0.0, xf])
            dy_f = np.diff(np.r_[0.0, yf])
            yaw_seg_f = np.arctan2(dy_f, dx_f)
            spd_f = np.hypot(dx_f, dy_f)

            Nf = len(xf)
            yaw_future = np.empty(Nf, float)

            # 첫 값은 "현재→+1" 구간의 방향(0을 강제하지 않음)
            if spd_f[0] < EPS:
                yaw_future[0] = 0.0   # 현재 헤딩이 0이므로 정지라면 0으로 대체
            else:
                yaw_future[0] = yaw_seg_f[0]

            # 이후는 연속각 유지하며 전파
            for i in range(1, Nf):
                if spd_f[i] < EPS:
                    yaw_future[i] = yaw_future[i-1]
                else:
                    prev = yaw_future[i-1]
                    yaw_future[i] = prev + wrap_to_pi(yaw_seg_f[i] - prev)
            # if idx == 200: import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            data['metadata']['past_states']['yaw'] = yaw_past.tolist()
            data['metadata']['future_states']['yaw'] = yaw_future.tolist()
            
            out_dir = out / seq_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{jf.stem}_heading.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Processed {jf} -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="root 디렉토리")
    parser.add_argument("--out", required=True, help="출력 디렉토리")
    parser.add_argument("--pattern", default="*_annotation.json", help="JSON 파일 패턴")
    args = parser.parse_args()

    main(args.root, args.out, args.pattern)
