import os
import json
import copy
import tempfile
from typing import Dict, List

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDatasetWaymo
import pyquaternion
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .vad_custom_nuscenes_eval import NuScenesEval_custom
from nuscenes.eval.common.utils import center_distance
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
from mmdet3d.core import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box as NuScenesBox
from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import CustomNuscenesBox
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from mmdet.datasets.pipelines import to_tensor
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.detection.constants import DETECTION_NAMES
import math
from ipdb import set_trace

from .waymo_dataset_sampling import sample_even_count_near_interval


camera_type = [
'front',
'front_right',
'front_left',
'rear',
'rear_left',
'rear_right',
'side_right',
'side_left'
]

@DATASETS.register_module()
class CustomWaymoDataset_new_subsampling_251014(NuScenesDatasetWaymo):
    r"""Custom NuScenes Dataset.
    """
    MAPCLASSES = ('divider',)
    def __init__(
        self,
        queue_length=4,
        bev_size=(200, 200),
        overlap_test=False,
        with_attr=True,
        fut_ts=6,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        map_classes=None,
        map_ann_file=None,
        map_fixed_ptsnum_per_line=-1,
        map_eval_use_same_gt_sample_num_flag=False,
        padding_value=-10000,
        use_pkl_result=False,
        custom_eval_version='vad_nusc_detection_cvpr_2019',
        use_video_test_mode=False,
        load_gt_map=False,
        seq_mode=False,
        seq_split_num=2,
        seq_length=3,
        filter_empty_gt=False,
        interval_2frames=True,
        test_mode=True,
        data_root = './data',
        with_heading=False, # set to true if you need heading
        heading_mask=False,
        cluster_data_dir=None,
        skip_cam=False,
        subsample_file=None,
        absolute_path=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.with_attr = with_attr
        self.fut_ts = fut_ts
        self.use_pkl_result = use_pkl_result
        self.filter_empty_gt = filter_empty_gt
        self.with_heading = with_heading
        self.absolute_path = absolute_path
        self.heading_mask = heading_mask
        
        self.padding_value = padding_value
        self.fixed_num = map_fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = map_eval_use_same_gt_sample_num_flag
        self.is_vis_on_test = False

        self.use_video_test_mode = use_video_test_mode
        self.load_gt_map = load_gt_map

        self.seq_mode = seq_mode
        self.seq_length = seq_length
        if seq_mode:
            self.seq_split_num = seq_split_num
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

        self.interval_2frames = interval_2frames
        
        self.test_mode = test_mode
        self.data_root = data_root
        
        if subsample_file is None:
            subsample_file = 'val_sample_heading_6000_stride2.txt' if self.test_mode \
                else 'train_sample_41566_stride2_longseq.txt'
        # the split list defines the exact subsample; prefer the copy pinned in the
        # repository so the sample set is identical regardless of what sits in data_root
        repo_split = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../../../tools/data_converter/waymo_e2e/splits', os.path.basename(subsample_file))
        if os.path.isabs(subsample_file) or os.path.exists(subsample_file):
            self.data_txt_path = subsample_file
        elif os.path.exists(self.data_root + subsample_file):
            self.data_txt_path = self.data_root + subsample_file
        elif os.path.exists(repo_split):
            self.data_txt_path = os.path.normpath(repo_split)
        else:
            raise FileNotFoundError(
                f'Waymo subsample list not found: tried {subsample_file}, '
                f'{self.data_root + subsample_file}, {os.path.normpath(repo_split)}')
        
        if self.test_mode:
            self.mode = 'val'
            self.data_path = self.data_root + 'waymo_e2e_val_6000'
        else:
            self.mode = 'train'
            self.data_path = self.data_root + 'waymo_e2e_training_41566'
        with open(self.data_txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.data_txt = [line.strip() for line in lines]

        ################# hh add ################
        self.sample_sequence_list = os.listdir(self.data_path)
        

        self.init_infos()
        
        if not self.test_mode:
            self._set_group_flag()
        
        ################# gw add ################
        # NOTE: Cluster 정보 로드   ------------------ # 
        self.skip_cam = skip_cam
        self.cluster_data_dir = cluster_data_dir
        if self.cluster_data_dir is not None:
            self.cluster_files = np.loadtxt(cluster_data_dir, dtype=float).tolist()
            if not self.cluster_files:
                raise FileNotFoundError(f"No cluster files found in: {cluster_data_dir}")
            self.cluster_infos = self.cluster_files
        # ------------------------------------------- #


    def init_infos(self):

        waymo_infos = []
        num_skipped_infos = 0
        
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            seq_dir = os.path.join(self.data_path, sequence_name)
            seq_paths = [f for f in os.listdir(seq_dir) if f.endswith(".json")]
            seq_paths = sorted(seq_paths)
            # seq_paths = seq_paths[1:-1] # first, last remove
            # # seq_paths = seq_paths[::16]  #15: 29172, 16: 27154
            for seq_path in seq_paths:
                info_path = os.path.join(seq_dir, seq_path)
                scene_name = os.path.join(sequence_name, seq_path)
                
                if not scene_name in self.data_txt:
                    continue

                if not os.path.exists(info_path):
                    num_skipped_infos += 1
                    continue
                # with open(info_path, 'rb') as f:
                #     infos = pickle.load(f)  
                with open(info_path, "r", encoding="utf-8") as f:
                    infos = json.load(f)
                waymo_infos.append(infos)
        self.data_infos = waymo_infos
        

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)


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
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        if self.load_gt_map:
            example = self.vectormap_pipeline(example,input_dict)
            if self.filter_empty_gt and \
                    ((example is None or ~(example['gt_labels_3d']._data != -1).any()) or \
                        (example is None or ~(example['map_gt_labels_3d']._data != -1).any())):
                return None
        else:
            if self.filter_empty_gt and \
                ((example is None or ~(example['gt_labels_3d']._data != -1).any())):
                return None

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
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)

                if self.load_gt_map:
                    example = self.vectormap_pipeline(example,input_dict)
                    if self.filter_empty_gt and \
                            ((example is None or ~(example['gt_labels_3d']._data != -1).any()) or \
                                (example is None or ~(example['map_gt_labels_3d']._data != -1).any())):
                        return None
                else:
                    if self.filter_empty_gt and \
                        ((example is None or ~(example['gt_labels_3d']._data != -1).any())):
                        return None

                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))

        if self.skip_cam:
            final_data = self.skip_union2one(data_queue)
        else:
            final_data = self.union2one(data_queue)
        # import pdb; pdb.set_trace()
        # final_data.pop('gt_bboxes_3d')
        return final_data

    def prepare_test_data_video(self, index):
        """Prepare data for testing as video sequence.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index and previous frames.
        """
        data_queue = []
        # set_trace()
        prev_indexs_list = list(range(index-self.queue_length, index))
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)

        input_dict = self.get_data_info(index)
        if input_dict is None:
            raise ValueError('input_dict is None')
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.is_vis_on_test:
            example = self.vectormap_pipeline(example, input_dict)
        
        data_queue.insert(0, example)
        for my_index, i in enumerate(prev_indexs_list):
            if self.interval_2frames and my_index != (len(prev_indexs_list) - 1):
                continue
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                raise ValueError('input_dict is None')
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.is_vis_on_test:
                    example = self.vectormap_pipeline(example,input_dict)
                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))
        return self.union2one(data_queue)


    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.is_vis_on_test:
            example = self.vectormap_pipeline(example, input_dict)
        return example

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
            
            if i == 0:
                metas_map[i]['prev_bev'] = False
                # prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # metas_map[i]['can_bus'][:3] = 0
                # metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                # tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # metas_map[i]['can_bus'][:3] -= prev_pos
                # metas_map[i]['can_bus'][-1] -= prev_angle
                # prev_pos = copy.deepcopy(tmp_pos)
                # prev_angle = copy.deepcopy(tmp_angle)

        #only update image, semantic_img, img_metas, others are only using last frame
        # queue[-1]['img'] = DC(torch.stack(imgs_list),
                            # cpu_only=False, stack=True)
        # if len(sem_imgs_list) > 0:
        #     queue[-1]['semantic_img'] = DC(torch.stack(sem_imgs_list),
        #                         cpu_only=False, stack=True)
                                
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue

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
            
            if i == 0:
                metas_map[i]['prev_bev'] = False
                # prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # metas_map[i]['can_bus'][:3] = 0
                # metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                # tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                # tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # metas_map[i]['can_bus'][:3] -= prev_pos
                # metas_map[i]['can_bus'][-1] -= prev_angle
                # prev_pos = copy.deepcopy(tmp_pos)
                # prev_angle = copy.deepcopy(tmp_angle)

        #only update image, semantic_img, img_metas, others are only using last frame
        queue[-1]['img'] = DC(torch.stack(imgs_list),
                            cpu_only=False, stack=True)
        # if len(sem_imgs_list) > 0:
        #     queue[-1]['semantic_img'] = DC(torch.stack(sem_imgs_list),
        #                         cpu_only=False, stack=True)
                                
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        if self.with_attr:
            gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_fut_masks = info['gt_agent_fut_masks'][mask]
            gt_fut_goal = info['gt_agent_fut_goal'][mask]
            gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels)

        return anns_results

    def get_his_traj(self, metainfo):
        x = np.array(metainfo["pos_x"])[1::2]   # shape (10,)
        y = np.array(metainfo["pos_y"])[1::2]   # shape (10,)

        # (N, 2) trajectory 생성
        traj = np.stack([-y, x], axis=1)   # shape (10, 2)

        # delta 계산 (인접 차이)
        deltas = traj[1:] - traj[:-1]     # shape (9, 2)

        # 필요하다면 앞의 8개만 잘라서 (8, 2)
        deltas = deltas[-2:]
        return deltas

    def get_his_yaw(self, metainfo):
        yaw = np.array(metainfo["yaw"])[1::2]
        
        # x = np.array(metainfo["pos_x"])[1::2]   # shape (10,)
        # y = np.array(metainfo["pos_y"])[1::2]   # shape (10,)

        # (N, 2) trajectory 생성
        # traj = np.stack([-y, x], axis=1)   # shape (10, 2)

        # delta 계산 (인접 차이)
        deltas = yaw[1:] - yaw[:-1]     # shape (9, 2)

        # 필요하다면 앞의 8개만 잘라서 (8, 2)
        deltas = deltas[-2:]
        return deltas
    
    def get_lcf_feat(self, metainfo):
        # x = np.array(metainfo["pos_x"])
        # y = np.array(metainfo["pos_y"])
        v_y = metainfo["vel_x"][-1]
        v_x = -metainfo["vel_y"][-1]
        a_x = metainfo["accel_x"][-1] 
        a_y = -metainfo["accel_y"][-1] 

        v0 = np.sqrt(v_x**2 + v_y**2)

        lcf_feat = np.array([v_x, v_y, a_x, a_y, v0])

        return lcf_feat

    def get_fut_traj(self, metainfo):
        x = np.array(metainfo["pos_x"])[1::2]   # shape (10,)
        y = np.array(metainfo["pos_y"])[1::2]   # shape (10,)

        # (N, 2) trajectory 생성
        traj = np.stack([-y, x], axis=1)   # shape (10, 2)

        # delta 계산 (인접 차이)
        deltas = traj[1:] - traj[:-1]     # shape (9, 2)

        # 필요하다면 앞의 8개만 잘라서 (8, 2)
        deltas = deltas[:8]
        return deltas
    
    def get_fut_yaw(self, metainfo):
        yaw = np.array(metainfo["yaw"])[1::2]
        
        # x = np.array(metainfo["pos_x"])[1::2]   # shape (10,)
        # y = np.array(metainfo["pos_y"])[1::2]   # shape (10,)

        # # (N, 2) trajectory 생성
        # traj = np.stack([-y, x], axis=1)   # shape (10, 2)

        # delta 계산 (인접 차이)
        deltas = yaw[1:] - yaw[:-1]     # shape (9, 2)

        # 필요하다면 앞의 8개만 잘라서 (8, 2)
        deltas = deltas[:8]
        return deltas
    
    def get_cmd(self, metainfo):
        if metainfo == 0:
            command = np.array([0, 0, 0, 1])  # Unknown
            # import pdb; pdb.set_trace()
            print("#######################unknown#######################")
        elif metainfo == 3:
            command = np.array([1, 0, 0, 0])  # Turn Right
        elif metainfo == 2:
            command = np.array([0, 1, 0, 0])  # Turn Left
        elif metainfo == 1:
            command = np.array([0, 0, 1, 0])  # Go Straight
        return command.astype(np.float32)
    
    def build_K(self, intrinsic_list):
        """[fu, fv, cu, cv, k1, k2, p1, p2, k3] -> 3x3 K (distortion 무시)"""
        fu, fv, cu, cv = intrinsic_list[:4]
        return np.array([[fu, 0.0, cu],
                        [0.0, fv, cv],
                        [0.0, 0.0, 1.0]], dtype=float)

    def get_v2c_from_cam2veh(self, extrinsic_list):
        """
        Waymo extrinsic: T_cam2veh (flattened 4x4) -> T_v2c (4x4 homogeneous)
        X_cam = R_v2c @ X_veh + t_v2c
        """
        T_cam2veh = np.array(extrinsic_list, dtype=float).reshape(4, 4)

        

        R_c2v = T_cam2veh[:3, :3]
        
        # R_c2v = np.matmul(axes_R, R_c2v)


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

        axes_R = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
            
        new_T_v2c = axes_tf @ T_v2c @ axes_R 
        return new_T_v2c

    def build_matrices(self, meta_calibration):
        """
        Return:
        - T_v2c (4x4): vehicle → camera 변환행렬
        - P (3x4): vehicle → image projection matrix (distortion 무시)
        """
        intrinsic_list, extrinsic_list = meta_calibration['intrinsic'], meta_calibration['extrinsic']
        K = self.build_K(intrinsic_list)
        T_v2c = self.get_v2c_from_cam2veh(extrinsic_list)
        Rt = T_v2c[:3, :]   # (3x4) [R|t]
        P = K @ Rt

        P_4x4 = np.zeros((4,4))
        P_4x4[:3,:4] = P
        P_4x4[3,3] = 1.0


        

        return T_v2c, P_4x4, K


    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
 
        if self.heading_mask:
            mask_ones=np.ones(len(self.get_fut_traj(info['metadata']['future_states'])), dtype=np.float32)
            mask_zeros=np.zeros(len(self.get_fut_traj(info['metadata']['future_states'])), dtype=np.float32)
            ego_fut_masks = np.stack([mask_ones, mask_ones, mask_zeros], axis=1)
        else:
            ego_fut_masks=np.ones(len(self.get_fut_traj(info['metadata']['future_states'])), dtype=np.float32)

 
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['sequence_id'],
            pts_filename=None,
            sweeps=None,
            ego2global_translation=None,
            ego2global_rotation=None,
            lidar2ego_translation=None,
            lidar2ego_rotation=None,
            prev_idx=int(info['frame_id']) - 1,
            next_idx=int(info['frame_id']) + 1,
            scene_token=info['sequence_id'],
            can_bus=None,
            frame_idx=int(info['frame_id']),
            timestamp=int(info['frame_id']) * 0.1,
            fut_valid_flag=True,
            map_location=None,
            ego_his_trajs=self.get_his_traj(info['metadata']['past_states']),
            ego_fut_trajs=self.get_fut_traj(info['metadata']['future_states']),
            # ego_fut_masks=np.ones(len(self.get_fut_traj(info['metadata']['future_states'])), dtype=np.float32),
            ego_fut_masks = ego_fut_masks,
            ego_fut_cmd=self.get_cmd(info['metadata']['intent']),
            # ego_lcf_feat=np.zeros(9).astype(np.float32) #info['gt_ego_lcf_feat']
            ego_lcf_feat = self.get_lcf_feat(info['metadata']['past_states'])
        )
        # lidar to ego transform
        
        if self.with_heading:
            # import pdb; pdb.set_trace()
            ego_fut_yaw = self.get_fut_yaw(info['metadata']['future_states'])
            ego_his_yaw = self.get_his_yaw(info['metadata']['past_states'])
            input_dict.update({
                'ego_his_trajs': np.concatenate([input_dict['ego_his_trajs'], np.expand_dims(ego_his_yaw, axis=1)], axis=1),
                'ego_fut_trajs': np.concatenate([input_dict['ego_fut_trajs'], np.expand_dims(ego_fut_yaw, axis=1)], axis=1), 
            })
            
        if self.absolute_path is True:
            _his = np.asarray(input_dict['ego_his_trajs'])
            input_dict['ego_his_trajs'] = np.vstack([
                np.zeros((1, _his.shape[1]), dtype=_his.dtype),
                np.cumsum(_his, axis=0),
            ])
            input_dict['ego_his_trajs'] = input_dict['ego_his_trajs'][::-1].copy()

            _fut = np.asarray(input_dict['ego_fut_trajs'])
            input_dict['ego_fut_trajs'] = np.cumsum(_fut, axis=0)
        # breakpoint()
        
        
        
        lidar2ego = np.eye(4).astype(np.float32)
        # lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        # lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego
        
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            input_dict["camera2ego"] = []
            input_dict["camera_intrinsics"] = []
            # for cam_type, cam_info in info['cams'].items():
            for cam_type in camera_type:
                data_path = os.path.join(self.data_path, info['sequence_id'], cam_type,  info['frame_id'] + '.jpg')
                image_paths.append(data_path)
                # obtain lidar to image transformation matrix
                lidar2cam_rt, lidar2img_rt, intrinsic = self.build_matrices(info['metadata']['camera_calibrations'][cam_type])

                # intrinsic = cam_info['cam_intrinsic']
                # viewpad = np.eye(4)
                # viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                # lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(intrinsic)
                lidar2cam_rts.append(lidar2cam_rt)
            
                # camera to ego transform
                # camera2ego = np.eye(4).astype(np.float32)
                # camera2ego[:3, :3] = Quaternion(
                #     cam_info["sensor2ego_rotation"]
                # ).rotation_matrix
                # camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
                # input_dict["camera2ego"].append(camera2ego)
                # # camera intrinsics
                # camera_intrinsics = np.eye(4).astype(np.float32)
                # camera_intrinsics[:3, :3] = cam_info["cam_intrinsic"]
                # input_dict["camera_intrinsics"].append(camera_intrinsics)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    camera_intrinsics=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        # NOTE: now we load gt in test_mode for evaluating
        # if not self.test_mode:
        #     annos = self.get_ann_info(index)
        #     input_dict['ann_info'] = annos

        # annos = self.get_ann_info(index)
        # input_dict['ann_info'] = annos

        # rotation = Quaternion(input_dict['ego2global_rotation'])
        # translation = input_dict['ego2global_translation']
        # can_bus = input_dict['can_bus']
        # can_bus[:3] = translation
        # can_bus[3:7] = rotation
        # patch_angle = quaternion_yaw(rotation) / np.pi * 180
        # if patch_angle < 0:
        #     patch_angle += 360
        # can_bus[-2] = patch_angle / 180 * np.pi
        # can_bus[-1] = patch_angle

        # lidar2ego = np.eye(4)
        # lidar2ego[:3,:3] = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
        # lidar2ego[:3, 3] = input_dict['lidar2ego_translation']
        # ego2global = np.eye(4)
        # ego2global[:3,:3] = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        # ego2global[:3, 3] = input_dict['ego2global_translation']
        # lidar2global = ego2global @ lidar2ego
        # input_dict['lidar2global'] = lidar2global

        # NOTE: 클러스터 정보 추가
        if self.cluster_data_dir is not None:
            current_cluster = self.cluster_infos[index]
            input_dict['cluster'] = torch.tensor(current_cluster, dtype=torch.long)
            
        input_dict['dataset_name'] = 'waymo'
        
        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        # import pdb; pdb.set_trace()
        # print('load idx', idx)
        if self.test_mode:
            return self.prepare_test_data_video(idx)
        
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            data['source']=torch.tensor([1])
            return data

    def _format_gt(self):
        gt_annos = []
        print('Start to convert gt map format...')
        # assert self.map_ann_file is not None
        if (not os.path.exists(self.map_ann_file)) :
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['token']
                gt_anno = {}
                gt_anno['sample_token'] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                gt_sample_dict = self.vectormap_pipeline(gt_sample_dict, self.data_infos[sample_id])
                gt_labels = gt_sample_dict['map_gt_labels_3d'].data.numpy()
                gt_vecs = gt_sample_dict['map_gt_bboxes_3d'].data.instance_list
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords)),
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno['vectors']=gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {
                'GTs': gt_annos
            }
            print('\n GT anns writes to', self.map_ann_file)
            mmcv.dump(nusc_submissions, self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')


    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        if isinstance(results, dict):
            # print(f'results must be a list, but get dict, keys={results.keys()}')
            # assert isinstance(results, list)
            results = results['bbox_results']
        assert isinstance(results, list)
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name == 'metric_results':
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         map_metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        detail = dict()
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=False)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.custom_eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']


        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import eval_map
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import format_res_gt_by_classes
        result_path = osp.abspath(result_path)
        
        print('Formating results & gts by classes')
        pred_results = mmcv.load(result_path)
        map_results = pred_results['map_results']
        gt_anns = mmcv.load(self.map_ann_file)
        map_annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     map_results,
                                                     map_annotations,
                                                     cls_names=self.MAPCLASSES,
                                                     num_pred_pts_per_instance=self.fixed_num,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
                                                     pc_range=self.pc_range)
        map_metrics = map_metric if isinstance(map_metric, list) else [map_metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in map_metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        for metric in map_metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)
            if metric == 'chamfer':
                thresholds = [0.5,1.0,1.5]
            elif metric == 'iou':
                thresholds= np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds),self.NUM_MAPCLASSES))
            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                map_results,
                                map_annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.MAPCLASSES,
                                logger=logger,
                                num_pred_pts_per_instance=self.fixed_num,
                                pc_range=self.pc_range,
                                metric=metric)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']
            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['NuscMap_{}/{}_AP'.format(metric,name)] =  cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()
            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]

        return detail
    
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
        # NOTE: print planning metric
        print('\n')
        print('-------------- Planning --------------')
        metric_dict = None
        num_valid = 0
        if not isinstance(results, list):
            results = results['bbox_results']
        # print(results)
        for res in results:
            if res['metric_results']['fut_valid_flag']:
                num_valid += 1
            else:
                continue

            if metric_dict is None:
                metric_dict = copy.deepcopy(res['metric_results'])
            else:
                for k in res['metric_results'].keys():
                    metric_dict[k] += res['metric_results'][k]
        
        for k in metric_dict:
            metric_dict[k] = metric_dict[k] / num_valid
            print("{}:{}".format(k, metric_dict[k]))
        #eval planning only
        # return metric_dict

        result_metric_names = ['EPA', 'ADE', 'FDE', 'MR']
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        all_metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                all_metric_dict[met+'_'+cls] = 0.0
        result_dict = {}
        for met in result_metric_names:
            for cls in motion_cls_names:
                result_dict[met+'_'+cls] = 0.0
        
        alpha = 0.5

        for i in range(len(results)):
            for key in all_metric_dict.keys():
                try:
                    all_metric_dict[key] += results[i]['metric_results'][key]
                except:
                    pass  # debug breakpoint removed for release
        for cls in motion_cls_names:
            # guard: planning-only evaluation has no detection/motion counts
            if all_metric_dict['gt_'+cls] > 0:
                result_dict['EPA_'+cls] = (all_metric_dict['hit_'+cls] - \
                     alpha * all_metric_dict['fp_'+cls]) / all_metric_dict['gt_'+cls]
            if all_metric_dict['cnt_ade_'+cls] > 0:
                result_dict['ADE_'+cls] = all_metric_dict['ADE_'+cls] / all_metric_dict['cnt_ade_'+cls]
            if all_metric_dict['cnt_fde_'+cls] > 0:
                result_dict['FDE_'+cls] = all_metric_dict['FDE_'+cls] / all_metric_dict['cnt_fde_'+cls]
                result_dict['MR_'+cls] = all_metric_dict['MR_'+cls] / all_metric_dict['cnt_fde_'+cls]
        
        print('\n')
        print('-------------- Motion Prediction --------------')
        for k, v in result_dict.items():
            print(f'{k}: {v}')

        # perception-free planning models emit no 3D boxes; the detection
        # formatting/evaluation below does not apply to them.
        if not (len(results) and isinstance(results[0], dict) and 'boxes_3d' in results[0]):
            return result_dict

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=metric, map_metric=map_metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric, map_metric=map_metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict
