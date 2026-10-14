from typing import Any, List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
# from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import _LRScheduler    # 기원 수정
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.law.transfuser_config import TransfuserConfig

# from navsim.agents.law.transfuser_model_v2 import V2TransfuserModel as TransfuserModel

from navsim.agents.law.transfuser_callback import TransfuserCallback 
from navsim.agents.law.transfuser_loss import transfuser_loss
from navsim.agents.law.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.law.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
import numpy as np
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import math

try:
    from mmcv.parallel import DataContainer  # mmcv 1.x
except Exception:
    DataContainer = None

def decontainerize_to_device(obj, device):
    """DataContainer/딕셔너리/리스트 안의 텐서를 재귀적으로 풀고 device로 이동."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)

    if DataContainer is not None and isinstance(obj, DataContainer):
        data = obj.data
        # collate([sample], samples_per_gpu=1)면 보통 길이 1
        if isinstance(data, (list, tuple)) and len(data) == 1:
            return decontainerize_to_device(data[0], device)
        return decontainerize_to_device(data, device)

    if isinstance(obj, dict):
        return {k: decontainerize_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(decontainerize_to_device(v, device) for v in obj)

    return obj  # 숫자/문자열/None 등

def get_device():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # 분산이면 LOCAL_RANK 우선, 아니면 current_device
        if "LOCAL_RANK" in os.environ:
            return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")

def to_numpy(x):
    # torch.Tensor일 수도 있으니 안전 변환
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        law_model_config = None,
        law_model_checkpoint = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path    
        
        # TODO: model 초기화 ###########################################
        # breakpoint()
        # self._transfuser_model = TransfuserModel(config)        # prev 방식    
        
        ################################################################
        # import sys
        # sys.path.append('')
        import sys
        prefix = law_model_config.split('/projects', 1)[0]
        sys.path.append(prefix)
        import numpy as np
        import argparse
        import mmcv
        import os
        import copy
        import torch
        torch.multiprocessing.set_sharing_strategy('file_system')
        import warnings
        from mmcv import Config, DictAction
        from mmcv.cnn import fuse_conv_bn
        from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
        from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                                wrap_fp16_model)

        # from mmdet3d.apis import single_gpu_test
        # from projects.mmdet3d_plugin.LAW.apis.test import single_gpu_test_custum        # 기원 추가
        # from mmdet3d.datasets import build_dataset
        # from projects.mmdet3d_plugin.datasets.builder import build_dataloader
        from mmdet3d.models import build_model
        from mmdet.apis import set_random_seed
        # from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test_for_law
        from mmdet.datasets import replace_ImageToTensor
        import time
        import os.path as osp
        import json
        import warnings
        warnings.filterwarnings("ignore")
        ######################################################################
        # breakpoint()  
        # TODO: model 체크포인트 및 config 설정 ###############################
        args_config = law_model_config
        args_cfg_options = None
        args_seed = 0
        args_deterministic = False
        args_checkpoint = law_model_checkpoint
        args_fuse_conv_bn = False
        ######################################################################
         
        cfg = Config.fromfile(args_config)
        if args_cfg_options is not None:
            cfg.merge_from_dict(args_cfg_options)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        # breakpoint()
        # import modules from plguin/xx, registry will be updated
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

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        # breakpoint()
        cfg.model.pretrained = None
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
        # breakpoint()
        distributed = False
        set_random_seed(args_seed, deterministic=args_deterministic)

        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        # breakpoint()        # model law model 잘 들고오는중
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        
        # breakpoint()
        checkpoint = load_checkpoint(model, args_checkpoint, map_location='cpu', strict=True)
        if args_fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE
        # breakpoint()
        model = MMDataParallel(model, device_ids=[0])
        
        
        # breakpoint()
        # outputs = single_gpu_test_custum(model, data_loader, args.show, args.show_dir)

        # breakpoint()
        self._transfuser_model = model      # 기존 model 대신 이걸 넣음

        self.pipeline = Compose(cfg.test_pipeline)
        self.cfg = cfg
        
        # TODO: model 초기화된거에서 pretrained law load -> 위에서 done
        # self.init_from_pretrained()                             # prev 방식




    def init_from_pretrained(self):
        # import ipdb; ipdb.set_trace()
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove 'agent.' prefix from keys if present
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
            
            # Load state dict and get info about missing and unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        
        # TODO: 여기서 모델에 넣어줄 때 전처리를 해주도록 하자
        
        # path = self.data_infos[idx]          # 경로 문자열
        # sample = self._load_pickle(path)     # 매 호출 시 디스크에서 로드
        # 필요하면 경로나 인덱스를 같이 반환
        # return {"data": sample, "meta": {"path": path, "idx": idx}}
        # breakpoint()
        dictA, dictB = features, targets
        merged_sample = {**dictA, **dictB}
        # TODO: 여기서 tuple을 dicionary로 반환
        # breakpoint()
        camera_order = ['f0', 'r0', 'l0', 'b0', 'l2', 'r2', 'r1', 'l1']  # waymo랑 통일해뒀음.
        lidar2img_mats = []
        padshape_list = []
        cur_img_path_list = []
        # breakpoint()

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
            cur_img_path_list.append(cur_img_path)
        # breakpoint()
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
        # breakpoint()
        # NOTE: (option 1) CUMSUM 기반으로 수행 (fut_traj만)
        if self.cfg.is_absolute_path is None or self.cfg.is_absolute_path == False:
            traj_abs_t = merged_sample['ego_fut_trajs']  # torch.Tensor
            zeros = torch.zeros_like(traj_abs_t[..., :1, :])                 # (..., 1, 2)
            prev  = torch.cat([zeros, traj_abs_t[..., :-1, :]], dim=-2)      # (..., T, 2)
            ego_fut_trajs_unifed_format = traj_abs_t - prev      
        else:
            ego_fut_trajs_unifed_format = merged_sample['ego_fut_trajs']
            
        if self.cfg.is_absolute_path is True:
            traj_his_abs_t = merged_sample['ego_his_trajs']  # 2,3 
            traj_his_abs_t_rev = torch.flip(traj_his_abs_t, dims=[0])   # [t - (t-1), (t-1) - (t-2)]
            rel_his_traj = -torch.cumsum(traj_his_abs_t_rev, dim=0)

            # 결과: [[t-1], [t-2]] 좌표 (기준은 t=0,0,0)
            rel_his_traj = torch.flip(rel_his_traj, dims=[0])   # 순서 복원: [t-2, t-1]
            zero = torch.zeros(1, rel_his_traj.size(1), device=rel_his_traj.device, dtype=rel_his_traj.dtype)
            his_traj_final = torch.cat([rel_his_traj, zero], dim=0)
        else: 
            his_traj_final = merged_sample['ego_his_trajs']
 
        raw_data = {
            # 'img_metas': img_metas,
            'scene_token': merged_sample['scene_token'],
            'frame_idx': merged_sample['frame_idx'],
            'lidar2img': lidar2img_array,
            'img_filename': cur_img_path_list,
            'ego_his_trajs': his_traj_final,
            'ego_fut_trajs': ego_fut_trajs_unifed_format,
            'ego_fut_masks': merged_sample['ego_fut_masks'],
            'ego_fut_cmd': ego_fut_cmd_unifed_format.unsqueeze(0),
            'ego_lcf_feat': merged_sample['ego_lcf_feat'],
            'gt_attr_labels': merged_sample['gt_attr_labels'],
            # 'aug_random_int': aug_random_int,
        }
        # breakpoint()
        
        ### 재영 선배가 맞춰주신 좌표계 여기도 반영 TODO: test에서도 반영되도록 하기 --------- ###
        T_neg90 = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        raw_data['lidar2img'] = np.matmul(raw_data['lidar2img'], T_neg90)
        R = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32)
        raw_data['ego_his_trajs'][:, :2] = raw_data['ego_his_trajs'][:, :2].type(torch.float32) @ R.T
        raw_data['ego_fut_trajs'][:, :2] = raw_data['ego_fut_trajs'][:, :2].type(torch.float32) @ R.T
        lcf = raw_data['ego_lcf_feat']
        raw_data['ego_lcf_feat'] = torch.tensor([-lcf[1], lcf[0], -lcf[3], lcf[2], lcf[4]])
        
        # import pdb; pdb.set_trace()
        self.with_heading = True
        if self.with_heading:
            if self.cfg.is_absolute_path is True:
                raw_data['ego_fut_trajs'][:, 2] += math.pi / 2 
                raw_data['ego_his_trajs'][:, 2] += math.pi / 2
        
        ### ------------------------------------- ###
        raw_data['isabsolute_path'] = self.cfg.is_absolute_path
        data = self.pipeline(raw_data)
        # breakpoint()
                
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [torch.cuda.current_device()])[0]
        result = self._transfuser_model(return_loss=False, rescale=True, navsim_inference=True, is_absolute_path=self.cfg.is_absolute_path, **data)
        '''
        # -------------------- 여기부터 교체 --------------------
        device = get_device()

        # 모델을 먼저 디바이스로
        self._transfuser_model = self._transfuser_model.to(device)

        # 배치 만들고 디바이스로 옮기기
        data = collate([data], samples_per_gpu=1)

        if device.type == "cuda":
            # 멀티/싱글 GPU 모두에서 안전: 인덱스(int)로 전달
            from mmcv.parallel import scatter
            dev_idx = device.index if device.index is not None else 0
            data = scatter(data, [dev_idx])[0]
        else:
            # CPU fallback: DataContainer 풀고 텐서만 CPU로
            data = decontainerize_to_device(data, device)

        with torch.no_grad():
            result = self._transfuser_model(
                return_loss=False, rescale=True, navsim_inference=True, **data
            )
        # -------------------- 교체 끝 --------------------
        '''
        # breakpoint()
        
        return result
        
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:        # 기원수정
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(type=self._config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self._config.weight_decay,
                            paramwise_cfg=self._config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self._config.scheduler_type,
                            milestones=self._config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._transfuser_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._transfuser_model.parameters()
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config)]
