import os, glob, pickle
import numpy as np
from torch.utils.data import Dataset
from mmdet.datasets import DATASETS
import torch
from mmdet.datasets.pipelines import Compose
import copy
from mmcv.parallel import DataContainer as DC
from pathlib import Path
import math
import os

def to_numpy(x):
    # torch.Tensor일 수도 있으니 안전 변환
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
@DATASETS.register_module()
class NavsimDataset_GW(Dataset):
    def __init__(self, data_dir, absolute_path=None, data_root=None, class_names=None, queue_length=None, seq_mode=False, pipeline=None, with_heading=True, cluster_data_dir=None, skip_cam=False, *args, **kwargs):
        # 1) pkl 경로만 수집
        self.pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
        if not self.pkl_files:
            raise FileNotFoundError(f"No .pkl files found in: {data_dir}")

        # 2) 경로만 보관 (메모리 사용 최소화)
        self.data_infos = self.pkl_files
        self.queue_length = queue_length
        self.with_heading = with_heading
        # 3) MMDet 호환 필드
        # TODO: flag 설정 해주기
        self.flag = np.zeros(len(self.pkl_files), dtype=np.int64)
        self.CLASSES = class_names
        self.seq_mode = seq_mode
        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self.load_gt_map = False
        self.filter_empty_gt = False
        self.interval_2frames = True
        self.data_root = data_root
        self.absolute_path = absolute_path
        
        # NOTE: Cluster 정보 로드   ------------------ # 
        self.cluster_data_dir = cluster_data_dir
        if self.cluster_data_dir is not None:
            self.cluster_files = np.loadtxt(cluster_data_dir, dtype=float).tolist()
            if not self.cluster_files:
                raise FileNotFoundError(f"No cluster files found in: {cluster_data_dir}")
            self.cluster_infos = self.cluster_files
        # ------------------------------------------- #

        self.skip_cam = skip_cam

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        data = self.prepare_train_data(idx)
        return data

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []

        # temporal aug
        prev_indexs_list = list(range(index-self.queue_length, index))
        # No More Shuffle!!
        # random.shuffle(prev_indexs_list) 
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        ##

        input_dict = self.get_data_info(index)
        if input_dict is None:
            raise ValueError('input_dict is None')
            return None

        aug_random_int = np.random.randint(0, 100000)
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        input_dict['aug_random_int'] = aug_random_int
        # self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        # if self.load_gt_map:
        #     example = self.vectormap_pipeline(example,input_dict)
        #     if self.filter_empty_gt and \
        #             ((example is None or ~(example['gt_labels_3d']._data != -1).any()) or \
        #                 (example is None or ~(example['map_gt_labels_3d']._data != -1).any())):
        #         return None
        # else:
        #     if self.filter_empty_gt and \
        #         ((example is None or ~(example['gt_labels_3d']._data != -1).any())):
        #         return None

        data_queue.insert(0, example)
        for my_index, i in enumerate(prev_indexs_list):
            if self.interval_2frames and my_index != (len(prev_indexs_list) - 1):
                continue
                
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                input_dict['aug_random_int'] = aug_random_int
                # self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)

                # if self.load_gt_map:
                #     example = self.vectormap_pipeline(example,input_dict)
                #     if self.filter_empty_gt and \
                #             ((example is None or ~(example['gt_labels_3d']._data != -1).any()) or \
                #                 (example is None or ~(example['map_gt_labels_3d']._data != -1).any())):
                #         return None
                # else:
                #     if self.filter_empty_gt and \
                #         ((example is None or ~(example['gt_labels_3d']._data != -1).any())):
                #         return None

                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))
        if self.skip_cam:
            final_data = self.skip_union2one(data_queue)
        else:
            final_data = self.union2one(data_queue)
        
        return final_data

    def get_data_info(self, idx):
        path = self.data_infos[idx]          # 경로 문자열
        sample = self._load_pickle(path)     # 매 호출 시 디스크에서 로드
        # 필요하면 경로나 인덱스를 같이 반환
        # return {"data": sample, "meta": {"path": path, "idx": idx}}
        # breakpoint()
        dictA, dictB = sample
        merged_sample = {**dictA, **dictB}
        # TODO: 여기서 tuple을 dicionary로 반환
        # breakpoint()
        # camera_order = ['f0', 'r1', 'l1', 'b0', 'l3', 'r3', 'r2', 'l2']  # waymo랑 통일해뒀음.
        camera_order = ['f0', 'r0', 'l0', 'b0', 'l2', 'r2', 'r1', 'l1']  # waymo랑 통일해뒀음.
        lidar2img_mats = []
        padshape_list = []
        cur_img_path_list = []

        # if not self.skip_cam:
        for cam in camera_order:
            if cam not in merged_sample['img_metas_tmp_imgpart']['lidar2img_tmp']:
                continue

            v = merged_sample['img_metas_tmp_imgpart']['lidar2img_tmp'][cam]

            R_c2l = to_numpy(v['sensor2lidar_rotation']).astype(np.float32)   # (3,3) cam->lidar
            t_c2l = to_numpy(v['sensor2lidar_translation']).astype(np.float32) # (3,)
            K     = to_numpy(v['intrinsics']).astype(np.float32)               # (3,3)

            # lidar -> camera (역변환)
            R_l2c = R_c2l.T                                     # (3,3)
            t_l2c = (-R_l2c @ t_c2l.reshape(3,)).reshape(3, 1)  # (3,1)

            # 4x4 외부행렬 T_l2c
            T_l2c = np.eye(4, dtype=np.float32)
            T_l2c[:3, :3] = R_l2c
            T_l2c[:3,  3] = t_l2c[:, 0]

            # 4x4 내부행렬 K4  (상단 3x3은 K, 마지막 열은 0, 마지막 행 [0,0,0,1])
            K4 = np.eye(4, dtype=np.float32)
            K4[0,0] = K[0,0];  K4[0,2] = K[0,2]
            K4[1,1] = K[1,1];  K4[1,2] = K[1,2]
            K4[2,2] = 1.0
            # 나머지 K4[0,1], K4[1,0], K4[0,3], K4[1,3], K4[2,0], K4[2,1], K4[2,3]는 0 유지

            # 4x4 lidar2img
            lidar2img_4x4 = K4 @ T_l2c                          # (4,4)
            lidar2img_mats.append(lidar2img_4x4)

            # pad_shape / 이미지 경로 등 나머지 처리
            cur_padshape = merged_sample['img_metas_tmp_imgpart']['pad_shape_tmp'][cam]
            padshape_list.append(cur_padshape)

            cur_img_path = merged_sample['img'][cam]    
            # TODO: 상대 경로로 바꾸기 ##################################
            # remap the cached absolute image path onto <data_root>/sensor_blobs/...
            p = Path(cur_img_path)
            parts = p.parts
            # use a separate name: `idx` is get_data_info's sample index and is
            # used below to look up this sample's cluster label
            sb_idx = parts.index("sensor_blobs")
            rel = Path(*parts[sb_idx:])
            cur_img_path = Path(f"{self.data_root}") / rel
            cur_img_path_list.append((cur_img_path))

        # (N, 3, 4) numpy array로 변환
        lidar2img_array = np.stack(lidar2img_mats, axis=0)          # TODO: 이게 img_metas에 들어갈 최종 mtx
        padshape_array = np.stack(padshape_list, axis=0)       
        # breakpoint()
        # img_metas = {
        #     'pad_shape': padshape_array,
        #     'lidar2img': lidar2img_array,
        #     'ego_fut_cmd': merged_sample['ego_fut_cmd'],
        #     'ego_fut_trajs': merged_sample['ego_fut_trajs'],
        #     'ego_fut_masks': merged_sample['ego_fut_masks'],
        # }
        """
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        lidar2img = torch.from_numpy(np.stack(img_metas[0]['lidar2img'])).to(img_feats.device).float()
        ego_cmd = img_metas[0]['ego_fut_cmd'].to(img_feat.device)[0, 0]
        gt_ego_fut_trajs = img_metas[0]['ego_fut_trajs'].to(img_feats.device)
        gt_ego_fut_masks = img_metas[0]['ego_fut_masks'].squeeze(0).unsqueeze(-1).to(img_feats.device)
        img_metas에서 필요로 하는 정보들 (for law)
        """


        '''
        < 공통 포맷 >
        turn right: 1 0 0 0 
        turn left: 0 1 0 0
        straight: 0 0 1 0
        unknown: 0 0 0 1
        
        < 현재 navsim 포맷 >
        turn right: 0 0 1 0 
        turn left: 1 0 0 0
        straight: 0 1 0 0
        unknown: 0 0 0 1
        '''
        # NOTE: 공통 format으로 변경
        perm = torch.tensor([2, 0, 1, 3], device=merged_sample['ego_fut_cmd'].device)
        ego_fut_cmd_unifed_format = merged_sample['ego_fut_cmd'].index_select(dim=-1, index=perm)  # 또는 x[..., [2,0,1,3]]
 
        # NOTE: (option 1) CUMSUM 기반으로 수행 (fut_traj만)
        if self.absolute_path is None or self.absolute_path == False:
            traj_abs_t = merged_sample['ego_fut_trajs']  # torch.Tensor
            zeros = torch.zeros_like(traj_abs_t[..., :1, :])                 # (..., 1, 2)
            prev  = torch.cat([zeros, traj_abs_t[..., :-1, :]], dim=-2)      # (..., T, 2)
            ego_fut_trajs_unifed_format = traj_abs_t - prev      
        else:
            ego_fut_trajs_unifed_format = merged_sample['ego_fut_trajs']
            
        if self.absolute_path is True:
            traj_his_abs_t = merged_sample['ego_his_trajs']  # 2,3 
            traj_his_abs_t_rev = torch.flip(traj_his_abs_t, dims=[0])   # [t - (t-1), (t-1) - (t-2)]
            rel_his_traj = -torch.cumsum(traj_his_abs_t_rev, dim=0)

            # 결과: [[t-1], [t-2]] 좌표 (기준은 t=0,0,0)
            rel_his_traj = torch.flip(rel_his_traj, dims=[0])   # 순서 복원: [t-2, t-1]
            zero = torch.zeros(1, rel_his_traj.size(1), device=rel_his_traj.device, dtype=rel_his_traj.dtype)
            his_traj_final = torch.cat([rel_his_traj, zero], dim=0)
        else: 
            his_traj_final = merged_sample['ego_his_trajs']
        # breakpoint()
        '''
        # NOTE: (option 2) 절대경로 기반으로 수행 
        ego_fut_trajs_unifed_format = merged_sample['ego_fut_trajs']
        '''
        # breakpoint()
        if self.with_heading:
            results = {
            # 'img_metas': img_metas,
            'scene_token': merged_sample['scene_token'],
            'frame_idx': merged_sample['frame_idx'],
            'sample_idx': merged_sample['frame_idx'],
            'lidar2img': lidar2img_array,
            'img_filename': cur_img_path_list,
            'ego_his_trajs': his_traj_final,
            'ego_fut_trajs': ego_fut_trajs_unifed_format,
            'ego_fut_masks': merged_sample['ego_fut_masks'],
            'ego_fut_cmd': ego_fut_cmd_unifed_format,
            'ego_lcf_feat': merged_sample['ego_lcf_feat'],
            'gt_attr_labels': merged_sample['gt_attr_labels'],
            # 'aug_random_int': aug_random_int,
            }
        else:
            results = {
                # 'img_metas': img_metas,
                'scene_token': merged_sample['scene_token'],
                'frame_idx': merged_sample['frame_idx'],
                'sample_idx': merged_sample['frame_idx'],
                'lidar2img': lidar2img_array,
                'img_filename': cur_img_path_list,
                'ego_his_trajs': his_traj_final[:, :2],
                'ego_fut_trajs': ego_fut_trajs_unifed_format[:, :2],
                'ego_fut_masks': merged_sample['ego_fut_masks'],
                'ego_fut_cmd': ego_fut_cmd_unifed_format,
                'ego_lcf_feat': merged_sample['ego_lcf_feat'],
                'gt_attr_labels': merged_sample['gt_attr_labels'],
                # 'aug_random_int': aug_random_int,
            }
        
        # if self.pipeline is not None:
        #     results = self.pipeline(results)   # ★ 파이프라인 적용
        # -90도 회전 (z축 기준)
        T_neg90 = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        results['lidar2img'] = np.matmul(results['lidar2img'], T_neg90)
        R = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32)
        results['ego_his_trajs'][:, :2] = results['ego_his_trajs'][:, :2].type(torch.float32) @ R.T
        results['ego_fut_trajs'][:, :2] = results['ego_fut_trajs'][:, :2].type(torch.float32) @ R.T
        lcf = results['ego_lcf_feat']
        results['ego_lcf_feat'] = torch.tensor([-lcf[1], lcf[0], -lcf[3], lcf[2], lcf[4]])
        
        # import pdb; pdb.set_trace()
        
        if self.with_heading:
            if  self.absolute_path is True:
                results['ego_fut_trajs'][:, 2] += math.pi / 2 
                results['ego_his_trajs'][:, 2] += math.pi / 2
        
        # NOTE: 클러스터 정보 추가
        if self.cluster_data_dir is not None:
            current_cluster = self.cluster_infos[idx]
            results['cluster'] = torch.tensor(current_cluster, dtype=torch.long)
            # breakpoint()
        
        results['dataset_name'] = 'navsim'
        results['is_absolute_path'] = self.absolute_path
            
        return results

    # ---- 내부 로더 ----
    def _load_pickle(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)





    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = []
        sem_imgs_list = []
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            single_img = each['img'][0].data if isinstance(each['img'], list) else each['img'].data
            imgs_list.append(single_img)
            if 'semantic_img' in each.keys():
                single_sem_img = each['semantic_img'][0].data if isinstance(each['semantic_img'], list) else each['semantic_img'].data
                sem_imgs_list.append(single_sem_img)

            metas_map[i] = each['img_metas'][0].data if isinstance(each['img_metas'], list) else each['img_metas'].data

            # include ego info into img_metas
            metas_map[i].update({
                'ego_fut_trajs': each['ego_fut_trajs'].data if not isinstance(each['ego_fut_trajs'], list) else each['ego_fut_trajs'][0].data,
                'ego_fut_masks': each['ego_fut_masks'].data if not isinstance(each['ego_fut_masks'], list) else each['ego_fut_masks'][0].data,
                'ego_fut_cmd': each['ego_fut_cmd'].data if not isinstance(each['ego_fut_cmd'], list) else each['ego_fut_cmd'][0].data,
                'ego_his_trajs': each['ego_his_trajs'].data if not isinstance(each['ego_his_trajs'], list) else each['ego_his_trajs'][0].data,
                'ego_lcf_feat': each['ego_lcf_feat'].data if not isinstance(each['ego_lcf_feat'], list) else each['ego_lcf_feat'][0].data,
                # 'cluster': each['cluster'].data if not isinstance(each['cluster'], list) else each['cluster'][0].data,
            })
            
            # if i == 0:
            #     metas_map[i]['prev_bev'] = False
            #     prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
            #     prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
            #     metas_map[i]['can_bus'][:3] = 0
            #     metas_map[i]['can_bus'][-1] = 0
            # else:
            #     metas_map[i]['prev_bev'] = True
            #     tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
            #     tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
            #     metas_map[i]['can_bus'][:3] -= prev_pos
            #     metas_map[i]['can_bus'][-1] -= prev_angle
            #     prev_pos = copy.deepcopy(tmp_pos)
            #     prev_angle = copy.deepcopy(tmp_angle)

        #only update image, semantic_img, img_metas, others are only using last frame
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                            cpu_only=False, stack=True)
        if len(sem_imgs_list) > 0:
            queue[-1]['semantic_img'] = DC(torch.stack(sem_imgs_list),
                                cpu_only=False, stack=True)
                                
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue
    
    def skip_union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = []
        sem_imgs_list = []
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            # single_img = each['img'][0].data if isinstance(each['img'], list) else each['img'].data
            # imgs_list.append(single_img)
            # if 'semantic_img' in each.keys():
            #     single_sem_img = each['semantic_img'][0].data if isinstance(each['semantic_img'], list) else each['semantic_img'].data
            #     sem_imgs_list.append(single_sem_img)

            metas_map[i] = each['img_metas'][0].data if isinstance(each['img_metas'], list) else each['img_metas'].data

            # include ego info into img_metas
            metas_map[i].update({
                'ego_fut_trajs': each['ego_fut_trajs'].data if not isinstance(each['ego_fut_trajs'], list) else each['ego_fut_trajs'][0].data,
                'ego_fut_masks': each['ego_fut_masks'].data if not isinstance(each['ego_fut_masks'], list) else each['ego_fut_masks'][0].data,
                'ego_fut_cmd': each['ego_fut_cmd'].data if not isinstance(each['ego_fut_cmd'], list) else each['ego_fut_cmd'][0].data,
                'ego_his_trajs': each['ego_his_trajs'].data if not isinstance(each['ego_his_trajs'], list) else each['ego_his_trajs'][0].data,
                'ego_lcf_feat': each['ego_lcf_feat'].data if not isinstance(each['ego_lcf_feat'], list) else each['ego_lcf_feat'][0].data,
                # 'cluster': each['cluster'].data if not isinstance(each['cluster'], list) else each['cluster'][0].data,
            })
            
            # if i == 0:
            #     metas_map[i]['prev_bev'] = False
            #     prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
            #     prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
            #     metas_map[i]['can_bus'][:3] = 0
            #     metas_map[i]['can_bus'][-1] = 0
            # else:
            #     metas_map[i]['prev_bev'] = True
            #     tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
            #     tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
            #     metas_map[i]['can_bus'][:3] -= prev_pos
            #     metas_map[i]['can_bus'][-1] -= prev_angle
            #     prev_pos = copy.deepcopy(tmp_pos)
            #     prev_angle = copy.deepcopy(tmp_angle)

        #only update image, semantic_img, img_metas, others are only using last frame
        # queue[-1]['img'] = DC(torch.stack(imgs_list),
        #                     cpu_only=False, stack=True)
        # if len(sem_imgs_list) > 0:
        #     queue[-1]['semantic_img'] = DC(torch.stack(sem_imgs_list),
        #                         cpu_only=False, stack=True)
                                
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue

    # ---- 시퀀스 예시 ----
    def get_sequence_data(self, idx, seq_len=5):
        k = seq_len // 2
        start = max(0, idx - k)
        end = min(len(self.pkl_files), idx + k + 1)
        indices = list(range(start, end))
        seq = [self._load_pickle(self.data_infos[i]) for i in indices]
        return {"indices": indices, "sequence": seq}


    def evaluate(self,
                 results,
                 metric='bbox',
                 map_metric='chamfer',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        if os.getenv("DEBUG_MODE") == "hm":
            #####추가 hm -> open loop metric 저장하기
            save_dir = os.path.join(self.work_dir if hasattr(self, "work_dir") else ".", "open_loop_metric_results/ours")
            os.makedirs(save_dir, exist_ok=True)

            # 파일명 충돌 방지용(필요하면 timestamp 추가)
            save_path = os.path.join(save_dir, "navsim_results_with_metrics.pkl")

            with open(save_path, "wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"[EVAL] dumped results -> {save_path}")
       
        # NOTE: print planning metric
        print('\n')
        print('-------------- Planning --------------')
        metric_dict = None
        num_valid = 0
        if not isinstance(results, list):
            results = results['bbox_results']
        # print(results)
        for res in results:
            # if res['metric_results']['fut_valid_flag']:
            #     num_valid += 1
            # else:
            #     continue
            num_valid += 1

            if metric_dict is None:
                metric_dict = copy.deepcopy(res['metric_results'])
            else:
                for k in res['metric_results'].keys():
                    metric_dict[k] += res['metric_results'][k]
        
        for k in metric_dict:
            metric_dict[k] = metric_dict[k] / num_valid
            print("{}:{}".format(k, metric_dict[k]))
        #eval planning only
        return metric_dict

        # result_metric_names = ['EPA', 'ADE', 'FDE', 'MR']
        # motion_cls_names = ['car', 'pedestrian']
        # motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
        #                        'fp', 'ADE', 'FDE', 'MR']
        # all_metric_dict = {}
        # for met in motion_metric_names:
        #     for cls in motion_cls_names:
        #         all_metric_dict[met+'_'+cls] = 0.0
        # result_dict = {}
        # for met in result_metric_names:
        #     for cls in motion_cls_names:
        #         result_dict[met+'_'+cls] = 0.0
        
        # alpha = 0.5

        # for i in range(len(results)):
        #     for key in all_metric_dict.keys():
        #         all_metric_dict[key] += results[i]['metric_results'][key]
        
        # for cls in motion_cls_names:
        #     result_dict['EPA_'+cls] = (all_metric_dict['hit_'+cls] - \
        #          alpha * all_metric_dict['fp_'+cls]) / all_metric_dict['gt_'+cls]
        #     result_dict['ADE_'+cls] = all_metric_dict['ADE_'+cls] / all_metric_dict['cnt_ade_'+cls]
        #     result_dict['FDE_'+cls] = all_metric_dict['FDE_'+cls] / all_metric_dict['cnt_fde_'+cls]
        #     result_dict['MR_'+cls] = all_metric_dict['MR_'+cls] / all_metric_dict['cnt_fde_'+cls]
        
        # print('\n')
        # print('-------------- Motion Prediction --------------')
        # for k, v in result_dict.items():
        #     print(f'{k}: {v}')

        # result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        # if isinstance(result_files, dict):
        #     results_dict = dict()
        #     for name in result_names:
        #         print('Evaluating bboxes of {}'.format(name))
        #         ret_dict = self._evaluate_single(result_files[name], metric=metric, map_metric=map_metric)
        #     results_dict.update(ret_dict)
        # elif isinstance(result_files, str):
        #     results_dict = self._evaluate_single(result_files, metric=metric, map_metric=map_metric)

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        # if show:
        #     self.show(results, out_dir, pipeline=pipeline)
        # return results_dict