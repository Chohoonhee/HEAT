import time
import copy
import numpy as np
import torch, cv2, os, random
import torch.nn as nn
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin import VAD_waymo
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric
from mmdet3d.models import builder

from projects.mmdet3d_plugin.LAW.utils import prj_pts_to_img, draw_lidar_pts, denormalize_img
import matplotlib.pyplot as plt
from ipdb import set_trace
import os.path as osp
import mmcv
from PIL import Image
from PIL import ImageDraw
import matplotlib

DATASET_SOURCE={
    0: 'nuScenes',
    1: 'Waymo',
    2: 'Navsim',
}

@DETECTORS.register_module()
class LAW_proto(VAD_waymo):
    def __init__(self,
                use_video=False,
                use_swin=False,
                only_front_view=False,
                use_multi_view=True,
                swin_input_channel=768,
                hidden_channel=256,
                use_semantic=False,
                semantic_img_backbone=None,
                flow_only=False,
                all_zero=False,
                semantic_only=False,
                use_2d_waypoint=False,
                wm_loss_weight=0.2,
                prototype_path = None,
                **kwargs,
                 ):
        self.view_mask = kwargs.pop('view_mask', slice(None, None))
        super().__init__( **kwargs)
        self.depth_eps = 2.75
        self.ref_pts_cam_list = []
        self.use_video = use_video
        self.use_swin = use_swin
        self.use_semantic = use_semantic

        self.flow_only = flow_only
        self.semantic_only = semantic_only
        self.all_zero = all_zero

        self.only_front_view = only_front_view
        self.use_2d_waypoint = use_2d_waypoint

        if semantic_img_backbone is not None:
            self.semantic_img_backbone = builder.build_backbone(semantic_img_backbone)

        if (not self.with_img_neck) and self.use_swin:
            self.swin_img_mlp = nn.Linear(swin_input_channel, hidden_channel)
        
        self.metrics_history = []
        self.call_count = 0
        self.wm_loss_weight = wm_loss_weight
        

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # My modification
            if self.use_video:
                img = img.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
                img_feats = self.img_backbone(img)
                img_feats = img_feats.mean(dim=2, keepdim=True)
            else:
                if self.use_swin:
                    img = img.unsqueeze(2)
                img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if not isinstance(img_feats, tuple):
            img_feats = [img_feats.squeeze(2)]

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        elif self.use_swin: #swin without fpn
            img_feats = [self.swin_img_mlp(img_feats[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()]

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def obtain_history_feat(self, imgs_queue, img_metas_list, is_test=False):
        """Obtain history BEV features iteratively."""
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
        img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)

        losses = {}
        pred_img_feat = None  # 마지막 값 리턴을 위해

        device = img_feats_list[0].device if hasattr(img_feats_list[0], 'device') else imgs_queue.device
        to32 = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)

        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list]
            img_feats = [each_scale[:, i] for each_scale in img_feats_list][0]
            pred_ego_fut_trajs, cur_img_feat, pred_img_feat = self.pts_bbox_head(img_feats, img_metas)

            if not is_test:
                # ------- (1) 배치 GT/MASK 수집 -------
                gts = []
                ms  = []
                for meta in img_metas:
                    gts.append(to32(meta['ego_fut_trajs']))
                    m = meta['ego_fut_masks']  # 보통 [1, T] 또는 [T]
                    m = to32(m)
                    # [B,1,T] 또는 [B,1,1,T] 같은 singleton을 정리해서 [T]로
                    while m.dim() >= 2 and m.size(0) == 1:
                        m = m.squeeze(0)
                    ms.append(m)

                # [B, ?, 2], [B, T]
                gt_ego_fut_trajs = torch.stack(gts, dim=0)  # safer than cat
                gt_ego_fut_masks = torch.stack(ms,  dim=0)

                # ------- (2) 모양/타입 표준화 (loss_waypoint와 동일) -------
                preds = pred_ego_fut_trajs.to(torch.float32)

                # gt: [B,T,2]
                gt = gt_ego_fut_trajs
                if gt.dim() == 4 and gt.size(1) == 1:
                    gt = gt[:, 0]
                elif gt.dim() != 3:
                    gt = gt.view(preds.size(0), -1, 2)

                # mask: [B,T,1], float32, [0,1] clamp
                m = gt_ego_fut_masks
                # [B,T] → [B,T,1]
                if m.dim() == 1:
                    m = m.unsqueeze(0)
                if m.dim() == 2:
                    m = m.unsqueeze(-1)
                elif m.dim() > 3:
                    m = m.view(preds.size(0), -1, 1)
                m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

                # 길이 맞추기
                T = min(preds.size(1), gt.size(1), m.size(1))
                preds, gt, m = preds[:, :T], gt[:, :T], m[:, :T]

                # ------- (3) 기존 loss 계산 + 안전 폴백 -------
                loss_prev_raw = self.pts_bbox_head.loss_3d(preds, gt, m)

                # 안전 L1 (마스크 합 기준 정규화)
                with torch.no_grad():
                    diff = (preds - gt).abs()
                    wsum = m.sum().clamp_(min=1.0)
                    loss_prev_safe = (diff * m).sum() / wsum

                loss_prev = loss_prev_safe if ((not torch.isfinite(loss_prev_raw)) or (loss_prev_raw.abs() > 1e6)) \
                                        else loss_prev_raw

                losses[f'prev_frame_loss_waypoint_{i}'] = loss_prev

        return losses, pred_img_feat


    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      semantic_img=None,
                      flow_img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None,
                      source=None
                      ):
        """
        agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
        """
        # if (img_metas[0][0]['cluster'].data != img_metas[0][1]['cluster'].data).item():
        #     import pdb; pdb.set_trace()
        # for s in source:
        #     print(DATASET_SOURCE.get(s.item(), 'Invalid'))
        
        if self.only_front_view:
            img = img[:, :, 0:1, ...]
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)

        self.pts_bbox_head.prev_view_feat = None

        if len_queue > 1:
            prev_frame_losses, pred_img_feat = self.obtain_history_feat(prev_img, prev_img_metas)  
        else:
            prev_frame_losses = {}

        cur_img = img[:, -1, ...]
        cur_img_metas = [each[len_queue-1] for each in img_metas]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        # import pdb; pdb.set_trace()
        # cur_img = cur_img.cpu() * std + mean
        # cur_img = torch.clamp(cur_img, 0, 1)
        # cv2.imwrite('cur_img.png', 255.0*cur_img[0][0].numpy().transpose(1,2,0))

        # prev_img = prev_img.cpu() * std + mean
        # prev_img = torch.clamp(prev_img, 0, 1)
        # cv2.imwrite('prev_img.png', 255.0*prev_img[0][0][0].numpy().transpose(1,2,0))
        

        cur_img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)[0]      
        losses = self.forward_pts_train(cur_img_feats, 
                                        cur_img_metas,
                                        pred_img_feat=pred_img_feat,
                                        ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                        ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                        ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels,
                                    )
        losses.update(prev_frame_losses)
        
        return losses

    def forward_pts_train(self,
                          img_feats,
                          img_metas,
                          pred_img_feat=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None,
                        ):
        """Forward function
        Args:
            ego_fut_cmd: [turn_left, turn_right, go_straight]
            ego_lcf_feat: (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度

        """
        #get the ego info   
        rank, _ = get_dist_info()

        # a) dtype/device/shape 표준화 (float32, [B,T,2]/[B,T,1])
        device = img_feats[0].device if isinstance(img_feats, (list,tuple)) else img_feats.device
        to32 = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)

        B = ego_his_trajs.size(0)
        ego_his_trajs = to32(ego_his_trajs).reshape(B, -1)
        ego_lcf_feat  = to32(ego_lcf_feat).reshape(B, -1)
        ego_fut_cmd   = to32(ego_fut_cmd).reshape(B, -1)
        ego_info = torch.cat([ego_his_trajs, ego_lcf_feat, ego_fut_cmd], dim=1)

        prev_pred_img_feat = pred_img_feat
        preds_ego_future_traj, cur_img_feat, pred_img_feat = self.pts_bbox_head(img_feats, img_metas, ego_info)
        # breakpoint()

        loss_rec = self.pts_bbox_head.loss_reconstruction(prev_pred_img_feat, cur_img_feat.detach())
        losses = {'loss_rec': loss_rec * self.wm_loss_weight}

        # ---- 표준화된 GT/Mask 만들기 ----
        gt = to32(ego_fut_trajs)
        if gt.dim()==4 and gt.size(1)==1: gt = gt[:,0]       # [B,T,2]
        if gt.dim()==3: pass
        else: gt = gt.view(B, -1, 2)

        m = to32(ego_fut_masks)
        # 다양한 케이스 → [B,T,1] 강제
        while m.dim()>=3 and m.size(1)==1: m = m.squeeze(1)
        if m.dim()==2: m = m.unsqueeze(-1)
        elif m.dim()>3: m = m.view(B, -1, 1)
        m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        preds = preds_ego_future_traj.to(torch.float32)

        T = min(preds.size(1), gt.size(1), m.size(1))
        preds, gt, m = preds[:, :T], gt[:, :T], m[:, :T]

        # b) 기존 loss 계산
        loss_waypoint_raw = self.pts_bbox_head.loss_3d(preds, gt, m)

        # c) 안전 L1(진단/대체)
        with torch.no_grad():
            diff = (preds - gt).abs()
            wsum = m.sum().clamp_(min=1.0)
            loss_safe = (diff * m).sum() / wsum

            if (not torch.isfinite(loss_waypoint_raw)) or (loss_waypoint_raw.abs()>1e6):
                sid = img_metas[0].get('sample_idx', 'unk')
                print(f'[rank{rank}] loss explode. raw={loss_waypoint_raw.item():.3e} safe={loss_safe.item():.3e} '
                    f'sid={sid} shapes P{tuple(preds.shape)} G{tuple(gt.shape)} M{tuple(m.shape)} '
                    f'Pmin/max=({preds.min().item():.3f},{preds.max().item():.3f}) '
                    f'Gmin/max=({gt.min().item():.3f},{gt.max().item():.3f}) Msum={m.sum().item():.1f}')

        loss_waypoint = loss_safe if ((not torch.isfinite(loss_waypoint_raw)) or (loss_waypoint_raw.abs()>1e6)) \
                                else loss_waypoint_raw

        losses['loss_waypoint'] = loss_waypoint
        return losses

    def simple_test_navsim(
        self,
        img_metas,
        img=None,
        # ego_his_trajs=None,
        # ego_fut_trajs=None,
        # ego_fut_cmd=None,
        # ego_lcf_feat=None,
        # gt_attr_labels=None, 
        is_absolute_path=None,
        **kwargs
    ):
        # breakpoint()
        # TODO: 여기서 shape를 맞춰주기 ##################################
        # 기존 evaluation 방법이랑 datashape 비교해서 해보자...  tmux 3번 참고 
        
        # img_metas = img_metas[0]
        # img = img[0]
        
        #################################################################
        # breakpoint()
        
        # len_queue = img.size(1)
        # prev_img = img[:, :-1, ...]
        # prev_img_metas = copy.deepcopy(img_metas)
        self.pts_bbox_head.prev_view_feat = None
        # if len_queue > 1:
        #     _ = self.obtain_history_feat(prev_img, prev_img_metas, is_test=True)  

        cur_img = img[0]
        cur_img_metas = img_metas[0]

        cur_img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)[0]  

        # bbox_list = [dict() for i in range(len(img_metas))]
        # B = img.size(0)
        # B = ego_his_trajs.size(0)
        # ego_his_trajs_ = ego_his_trajs.reshape(B, -1)
        # ego_lcf_feat_ = ego_lcf_feat.reshape(B, -1)
        # ego_fut_cmd_ = ego_fut_cmd.reshape(B, -1)
        # ego_info = torch.cat([ego_his_trajs_, ego_lcf_feat_, ego_fut_cmd_], dim=1)

        preds_ego_future_traj, _, _ = self.pts_bbox_head(
                                        cur_img_feats, 
                                        cur_img_metas, 
                                    )
        # breakpoint()
        with torch.no_grad():
            # pre-process
            # gt_bbox = gt_bboxes_3d[0][0]
            # gt_label = gt_labels_3d[0][0].to('cpu')
            # gt_attr_label = gt_attr_labels[0][0].to('cpu')
            # fut_valid_flag = bool(fut_valid_flag[0][0])

            # ego planning metric
            # assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = preds_ego_future_traj[0]
            # ego_fut_trajs = ego_fut_trajs[0, 0]
            # ego_fut_cmd = ego_fut_cmd[0, 0, 0]
        
        
        
        
        
            # TODO: 나중에 최종 law 학습 시 difference로 할 것
            if is_absolute_path == True:
                # breakpoint()
                pass
            elif is_absolute_path == False:
                ego_fut_preds = ego_fut_preds.cumsum(dim=-2)
            else:
                raise ValueError("Please input absolute_path")
        
        
            # ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            # metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
            #     pred_ego_fut_trajs = ego_fut_preds[None],
            #     gt_ego_fut_trajs = ego_fut_trajs[None],
            #     # gt_agent_boxes = gt_bbox,
            #     gt_agent_feats = gt_attr_label.unsqueeze(0),
            #     # fut_valid_flag = fut_valid_flag
            # )

            #mid print
            # update metrics
            # self.metrics_history.append(metric_dict_planner_stp3)
            # self.call_count += 1

            # print results
            # if self.call_count % 500 == 0:
            #     self.compute_and_print_metrics_average()
                
        return ego_fut_preds

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        is_absolute_path=None,
        **kwargs
    ):
        
        # if ego_fut_cmd[0][0][0][0][2] != 1:
            # breakpoint()
        # breakpoint()
        # img 위에, ego_fut_trajs (front cam)랑 ego_his_trajs (rear cam에)를 둘 다 viz 해주셈
        # 
        # img[:, :, self.view_mask] = 0.0
        # import pdb; pdb.set_trace()
        
        is_absolute_path = True

        if kwargs.get("navsim_inference", False):
            # breakpoint()
            bbox_results = self.simple_test_navsim(
                img_metas=img_metas,
                img=img,
                is_absolute_path=is_absolute_path,
                **kwargs
            )
        else:
            # breakpoint()
            bbox_results = self.simple_test(
                img_metas=img_metas,
                img=img,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                ego_his_trajs=ego_his_trajs[0],
                ego_fut_trajs=ego_fut_trajs[0],
                ego_fut_cmd=ego_fut_cmd[0],
                ego_lcf_feat=ego_lcf_feat[0],
                gt_attr_labels=gt_attr_labels,
                is_absolute_path=is_absolute_path,
                **kwargs
            )

        return bbox_results
    
    def simple_test(
        self,
        img_metas,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        fut_valid_flag=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        is_absolute_path=False,
        **kwargs,
    ):
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        
        prev_img_metas = copy.deepcopy(img_metas)
        self.pts_bbox_head.prev_view_feat = None
        if len_queue > 1:
            _ = self.obtain_history_feat(prev_img, prev_img_metas, is_test=True)  
        
        cur_img = img[:, -1, ...]
        cur_img_metas = [each[len_queue-1] for each in img_metas]

        

        cur_img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)[0]  

        bbox_list = [dict() for i in range(len(img_metas))]

        
        
        debug=False # DEFAULT: False # hj
        metric_dict = self.simple_test_pts(
            cur_img_feats,
            cur_img_metas,
            cur_img, # for vis
            gt_bboxes_3d,
            gt_labels_3d,
            fut_valid_flag=fut_valid_flag,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
            debug=debug,
            is_absolute_path=is_absolute_path,
        )
        if debug:
            ego_fut_preds = metric_dict
            img_metas = cur_img_metas
            # import pdb; pdb.set_trace()
        
        for result_dict in bbox_list:
            result_dict['metric_results'] = metric_dict

        return bbox_list
    
    # def simple_test_pts(
    #     self,
    #     img_feats,
    #     img_metas,
    #     gt_bboxes_3d=None,
    #     gt_labels_3d=None,
    #     fut_valid_flag=None,
    #     ego_his_trajs=None,
    #     ego_fut_trajs=None,
    #     ego_fut_cmd=None,
    #     ego_lcf_feat=None,
    #     gt_attr_labels=None,
    #     debug=False
    # ):
    #     """Test function"""
    #     B = ego_his_trajs.size(0)
    #     ego_his_trajs_ = ego_his_trajs.reshape(B, -1)
    #     ego_lcf_feat_ = ego_lcf_feat.reshape(B, -1)
    #     ego_fut_cmd_ = ego_fut_cmd.reshape(B, -1)
    #     ego_info = torch.cat([ego_his_trajs_, ego_lcf_feat_, ego_fut_cmd_], dim=1)

    #     # breakpoint()
    #     preds_ego_future_traj, _, _ = self.pts_bbox_head(
    #                                     img_feats, 
    #                                     img_metas, 
    #                                 )
    #     if debug:
    #         return preds_ego_future_traj
    #     with torch.no_grad():
    #         # pre-process
    #         if gt_bboxes_3d is not None:
    #             gt_bbox = gt_bboxes_3d[0][0]
    #             gt_label = gt_labels_3d[0][0].to('cpu')
    #             gt_attr_label = gt_attr_labels[0][0].to('cpu').unsqueeze(0)
    #         else:
    #             gt_bbox, gt_label, gt_attr_label = None, None, None
    #         if fut_valid_flag is not None:
    #             fut_valid_flag = bool(fut_valid_flag[0][0])
    #         else:
    #             fut_valid_flag = True

    #         # ego planning metric
    #         assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
    #         ego_fut_preds = preds_ego_future_traj[0]
    #         ego_fut_trajs = ego_fut_trajs[0, 0]
    #         ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            
    #         ego_fut_preds = ego_fut_preds.cumsum(dim=-2)[:, :2]
    #         ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)[:, :2]

    #         # if os.environ['DEBUG'] =='JY': import pdb; pdb.set_trace()
    #         metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
    #             pred_ego_fut_trajs = ego_fut_preds[None],
    #             gt_ego_fut_trajs = ego_fut_trajs[None],
    #             gt_agent_boxes = gt_bbox,
    #             gt_agent_feats = gt_attr_label,
    #             fut_valid_flag = fut_valid_flag
    #         )

    #         #mid print
    #         # update metrics
    #         self.metrics_history.append(metric_dict_planner_stp3)
    #         self.call_count += 1
    #         # breakpoint()

    #         # print results
    #         if self.call_count % 500 == 0:
    #             self.compute_and_print_metrics_average()

    #     return metric_dict_planner_stp3
    

    def simple_test_pts(
        self,
        img_feats,
        img_metas,
        cur_img, 
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        fut_valid_flag=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        debug=False,
        is_absolute_path=False
    ):
        """Test function"""
        B = ego_his_trajs.size(0)
        ego_his_trajs_ = ego_his_trajs.reshape(B, -1)
        ego_lcf_feat_ = ego_lcf_feat.reshape(B, -1)
        ego_fut_cmd_ = ego_fut_cmd.reshape(B, -1)
        ego_info = torch.cat([ego_his_trajs_, ego_lcf_feat_, ego_fut_cmd_], dim=1)

        # breakpoint()
        preds_ego_future_traj, _, _ = self.pts_bbox_head(
                                        img_feats, 
                                        img_metas, 
                                    )
        # visualize 
        debug_mode = os.getenv("DEBUG_MODE", "DEFAULT")
        if debug_mode =='HM':
            self.visualize_test(preds_ego_future_traj, ego_fut_trajs, img_metas, cur_img, is_absolute_path)
            # breakpoint()
        
        if debug:
            return preds_ego_future_traj
        with torch.no_grad():
            # pre-process
            if gt_bboxes_3d is not None:
                # breakpoint()
                # gt_bbox = gt_bboxes_3d[0][0]  # TODO: 기원 수정 251030
                gt_bbox = gt_bboxes_3d[0]
                # gt_label = gt_labels_3d[0][0].to('cpu')  # TODO: 기원 수정 251030
                gt_label = gt_labels_3d[0].to('cpu')  # TODO: 기원 수정 251030
                # gt_attr_label = gt_attr_labels[0][0].to('cpu').unsqueeze(0)   # TODO: 기원 수정 251030
                # breakpoint()
                gt_attr_label = gt_attr_labels[0].to('cpu').unsqueeze(0)
            else:
                gt_bbox, gt_label, gt_attr_label = None, None, None
            if fut_valid_flag is not None:
                # breakpoint()
                # breakpoint()
                # fut_valid_flag = bool(fut_valid_flag[0][0])       # TODO: 기원 수정 251030
                fut_valid_flag = bool(fut_valid_flag[0])
            else:
                fut_valid_flag = True

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = preds_ego_future_traj[0]
            # breakpoint()
            if ego_fut_trajs.dim() == 4: # TODO: 기원 수정 251030
                ego_fut_trajs = ego_fut_trajs[0, 0]
            elif ego_fut_trajs.dim() == 3: # TODO: 기원 수정 251030
                ego_fut_trajs = ego_fut_trajs[0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            
            # breakpoint()
            # breakpoint()
            if is_absolute_path == True:
                ego_fut_preds = ego_fut_preds[:, :2]
                ego_fut_trajs = ego_fut_trajs[:, :2]
            else:
                ego_fut_preds = ego_fut_preds.cumsum(dim=-2)[:, :2]
                ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)[:, :2]

            # if os.environ['DEBUG'] =='JY': import pdb; pdb.set_trace()
            # breakpoint()
            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_preds[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label,
                fut_valid_flag = fut_valid_flag
            )

            #mid print
            # update metrics
            self.metrics_history.append(metric_dict_planner_stp3)
            self.call_count += 1
            # breakpoint()

            # print results
            if self.call_count % 20 == 0:
                self.compute_and_print_metrics_average()

        return metric_dict_planner_stp3
    
    def visualize_test(self, pred_trajs, ego_fut_trajs, img_metas, cur_img, is_absolute_path=True):
        # breakpoint()
        cur_front_img_path = img_metas[0]['filename'][0]
        lidar2front_img = img_metas[0]['lidar2img'][0]
        
        if is_absolute_path:
            gt_fut_trajs = np.array(img_metas[0]['ego_fut_trajs'][0, :, :2])
        else:
            gt_fut_trajs = np.array(img_metas[0]['ego_fut_trajs'].cumsum(dim=-2)[0, :, :2])
        
        if pred_trajs.ndim == 3:
            # [num_mode, T, 2] -> 첫 번째 모드만 사용
            pred_fut_trajs = pred_trajs[0].detach().cpu().numpy()[:, :2]
        elif pred_trajs.ndim == 2:
            # [T, 2]
            pred_fut_trajs = pred_trajs.detach().cpu().numpy()[:, :2]

            # 5) 이미지 로드
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        # mean = torch.tensor(img_metas[0]['img_norm_cfg']['mean']).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        # std = torch.tensor(img_metas[0]['img_norm_cfg']['std']).view(3,1,1)

        
        # img = mmcv.imread(cur_front_img_path)  # BGR
        front_img = cur_img[0].detach().cpu()
        front_right_img = cur_img[1].detach().cpu()
        front_left_img = cur_img[2].detach().cpu()
        img_list = [front_img, front_right_img, front_left_img]
        result_img_list = []
        for i in range(3):
            current_front_img = img_list[i]
            current_front_img = current_front_img * std + mean
            current_front_img = torch.clamp(current_front_img, 0, 1)
            import torchvision.transforms as T
            to_pil = T.ToPILImage()
            current_front_img = to_pil(current_front_img)
            result_img_list.append(current_front_img)
                


        # breakpoint()
        # breakpoint()
        H, W = front_img.size()[1:]
        front_img = front_img * std + mean
        front_img = torch.clamp(front_img, 0, 1)
        import torchvision.transforms as T
        to_pil = T.ToPILImage()
        front_img = to_pil(front_img)
        
        

        # 6) GT / Pred 궤적 projection
        gt_pts_img = self.project_points_xy(gt_fut_trajs, lidar2front_img, (H, W))
        pred_pts_img = self.project_points_xy(pred_fut_trajs, lidar2front_img, (H, W))

        # BGR 색상
        # color_gt = (0, 255, 0)   # 초록
        # color_pred = (255, 0, 0) # 빨강
        # color_gt   = (74, 222, 128)   # 예쁜 연두
        # color_pred = (249, 115, 115)  # 코랄 레드

        # self.draw_traj(front_img, gt_pts_img, color_gt, (H, W), thickness=4)
        # self.draw_traj(front_img, pred_pts_img, color_pred, (H, W), thickness=4)

        # self.draw_traj_continuous(
        #         front_img,
        #         gt_pts_img,
        #         color_gt,
        #         (H, W),
        #         cmap=plt.cm.Greens,
        #         thickness=5,
        #         n_samples=80,   # 더 부드럽게 하고 싶으면 ↑
        #         # alpha=200, 
        # )

        # # Pred: 노랑→빨강 같은 플라즈마 계열 그라디언트
        # self.draw_traj_continuous(
        #     front_img,
        #     pred_pts_img,
        #     color_pred,
        #     (H, W),
        #     cmap=plt.cm.Reds,
        #     thickness=5,
        #     n_samples=80,
        #     alpha=180,
        # )


        self.draw_traj_dotted(
            front_img,
            gt_pts_img,
            (H, W),
            colormap='winter',
            traj_score=1.0,     # 0~1, 낮추면 더 하얗게 흐려짐
            points_per_step=120,  # segment당 찍을 점 개수
            radius=3,
            alpha=220,
        )

        # Pred: 'autumn' 계열 (노랑→빨강)
        self.draw_traj_dotted(
            front_img,
            pred_pts_img,
            (H, W),
            colormap='autumn',
            traj_score=1.0,
            points_per_step=120,
            radius=3,
            alpha=255,
        )
        T_sec = 4.0
        self.save_time_colorbar(
            cmap_name='autumn',
            t_min=0.0,
            t_max=T_sec,
            out_path='pred_time_colorbar.png',
            horizontal=True,
        )
        self.save_time_colorbar(
            cmap_name='winter',    # GT에 쓴 colormap
            t_min=0.0,
            t_max=T_sec,
            out_path='gt_time_colorbar.png',
            horizontal=True,
        )


        

        # 8) 저장 경로 만들기
        # breakpoint()
        save_root = 'waymo_openloop_vis_ours'
        # breakpoint()
        os.makedirs(save_root, exist_ok=True)
        base_name = osp.basename(cur_front_img_path)
        name, ext = osp.splitext(base_name)
        save_path = osp.join(save_root, cur_front_img_path.split('/')[-3] + f'_{name}_traj.png')
        save_right_path = osp.join(save_root, cur_front_img_path.split('/')[-3] + f'_{name}_front_right_traj.png')
        save_left_path = osp.join(save_root, cur_front_img_path.split('/')[-3] + f'_{name}_front_lefth_traj.png')
    
        front_img.save(save_path)
        result_img_list[1].save(save_right_path)
        result_img_list[2].save(save_left_path)

        


        print(f'Saved traj visualization to {save_path}')
        
        
        # h_gt_fut_trajs = torch.stack
    def compute_and_print_metrics_average(self):
        # compute avg
        avg_metrics = {key: sum(m[key] for m in self.metrics_history) / len(self.metrics_history) for key in self.metrics_history[0]}
        # print avg
        print(f"\n Average metrics after {len(self.metrics_history)} calls: {avg_metrics}")


    def project_points_xy(self, points_xy, lidar2img, img_shape):
        """
        points_xy: (N, 2) in LiDAR/ego coord, z=0으로 가정
        lidar2img: (4, 4)
        img_shape: (H, W, C)
        return: 이미지 안에 들어오는 2D 좌표 (M, 2)
        """
        H, W = img_shape
        N = points_xy.shape[0]

        # (N,3): [x, y, z=0]
        pts_3d = np.concatenate(
            [points_xy, (0.5)*np.ones((N, 1), dtype=points_xy.dtype)],
            axis=-1
        )  # (N, 3)

        # homogeneous (N,4)
        pts_3d_h = np.concatenate(
            [pts_3d, np.ones((N, 1), dtype=points_xy.dtype)],
            axis=-1
        )

        # (N,4) @ (4,4)^T -> (N,4)
        
        pts_2d_h = (lidar2img @ pts_3d_h.T).T

        # depth = z
        depth = pts_2d_h[:, 2]
        valid_depth = depth > 0

        pts_2d_h = pts_2d_h[valid_depth]
        depth = depth[valid_depth]

        if pts_2d_h.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # normalize by depth
        pts_2d = pts_2d_h[:, :2] /  np.clip(depth[:, None], 1e-6, None)  # (M,2)

        # 이미지 범위 안에 있는 점만 남기기
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        # valid_x = (x >= 0) & (x < W)
        # valid_y = (y >= 0) & (y < H)
        # valid = valid_x & valid_y

        # return pts_2d[valid]
        return pts_2d

    
    def draw_traj(self, img, pts, color, img_shape, radius=4, thickness=2):
        """
        img: PIL.Image
        pts: (N, 2) numpy float
        color: (R, G, B)
        """
        if pts.shape[0] == 0:
            return
        # breakpoint()
        H, W = img_shape
        draw = ImageDraw.Draw(img)

        # float → int
        pts_int = [(int(u), int(v)) for (u, v) in pts]

        # 선으로 연결
        if len(pts_int) >= 2:
            draw.line(pts_int, fill=color, width=thickness)

        # 점 찍기
        for (u, v) in pts_int:
            if not (0 <= u < W and 0 <= v < H):
                continue
            x0, y0 = u - radius, v - radius
            x1, y1 = u + radius, v + radius
            draw.ellipse([x0, y0, x1, y1], fill=color)
        
    
    def draw_traj_continuous(self,
                         img,
                         pts,
                         color,
                         img_shape,
                         thickness=6,
                         cmap=None,
                         alpha=255,
                         n_samples=80):
        """
        img      : PIL.Image
        pts      : (N, 2) float (u, v)   (LiDAR→img proj 결과)
        color    : (R,G,B)  # cmap=None일 때만 사용
        img_shape: (H, W)
        """

        def _resample_polyline(pts, n_samples=80):
            """거리 기준으로 균일하게 다시 샘플링해서 곡선을 부드럽게."""
            pts = np.asarray(pts, dtype=float)
            if pts.shape[0] < 2:
                return pts

            diff = np.diff(pts, axis=0)
            seg_len = np.sqrt((diff ** 2).sum(axis=1))          # 각 segment 길이
            cumdist = np.concatenate([[0.0], np.cumsum(seg_len)])
            total = cumdist[-1]
            if total == 0:
                return pts

            s = np.linspace(0.0, total, n_samples)
            new_pts = np.zeros((n_samples, 2), dtype=float)
            for d in range(2):
                new_pts[:, d] = np.interp(s, cumdist, pts[:, d])
            return new_pts

            if pts.shape[0] < 2:
                return

        H, W = img_shape
        # 1) 먼저 리샘플해서 곡선을 부드럽게
        pts_rs = _resample_polyline(pts, n_samples=n_samples)
        pts_int = [(int(u), int(v)) for (u, v) in pts_rs]

        # --- 단색 모드 (cmap 없이) ---
        if cmap is None:
            draw = ImageDraw.Draw(img)
            draw.line(pts_int, fill=color, width=thickness)
            return

        # --- 그라디언트 모드 ---
        draw = ImageDraw.Draw(img, "RGBA")

        # t = np.linspace(0.0, 1.0, len(pts_int))
        # cols = (cmap(t) * 255).astype(np.uint8)   # (N,4)
        t = np.linspace(0.0, 1.0, len(pts_int))
        cols = (cmap(t) * 255).astype(np.uint8)

        for i in range(len(pts_int) - 1):
            (x0, y0), (x1, y1) = pts_int[i], pts_int[i + 1]

            if not (0 <= x0 < W and 0 <= y0 < H):
                continue
            if not (0 <= x1 < W and 0 <= y1 < H):
                continue

            r, g, b, _ = cols[i]
            draw.line(
                [x0, y0, x1, y1],
                fill=(int(r), int(g), int(b), alpha),
                width=thickness,
            )
    

    def _render_traj_img(self,
                     pts_img,
                     traj_score=1.0,
                     colormap='winter',
                     points_per_step=5):
        """
        pts_img    : (T, 2) np.array, 이미 'image 좌표 (u,v)' 상태
        traj_score : [0,1], 1이면 원색, 0이면 하얗게 흐려짐
        colormap   : matplotlib colormap 이름 (e.g., 'winter', 'autumn', 'Reds', ...)
        """
        pts = np.asarray(pts_img, dtype=float)
        if pts.shape[0] < 2:
            return pts, None

        total_steps = (len(pts) - 1) * points_per_step + 1

        # 색상 생성 (0~1 → cmap → 0~255 RGB)
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps)
        )[:, :3] * 255.0

        # traj_score로 밝기 조절
        dot_colors = dot_colors * traj_score + \
            (1 - traj_score) * np.ones_like(dot_colors)

        # 이미지 좌표에서 직선 보간
        total_xy = np.zeros((total_steps, 2), dtype=float)
        for i in range(total_steps - 1):
            seg_idx = i // points_per_step      # 어느 구간에 속하는지
            p0 = pts[seg_idx]
            p1 = pts[seg_idx + 1]
            local_t = (i / points_per_step) - seg_idx
            total_xy[i] = p0 + local_t * (p1 - p0)

        total_xy[-1] = pts[-1]
        return total_xy, dot_colors


    def draw_traj_dotted(self,
                        img,
                        pts_img,
                        img_shape,
                        colormap='winter',
                        traj_score=1.0,
                        points_per_step=5,
                        radius=3,
                        alpha=255):
        """
        img       : PIL.Image
        pts_img   : (T, 2) float, '이미 proj된 이미지 좌표' (gt_pts_img, pred_pts_img)
        img_shape : (H, W)
        """

        H, W = img_shape

        dense_xy, dot_colors = self._render_traj_img(
            pts_img,
            traj_score=traj_score,
            colormap=colormap,
            points_per_step=points_per_step,
        )

        if dense_xy is None:
            return

        draw = ImageDraw.Draw(img, "RGBA")

        for (u, v), color in zip(dense_xy, dot_colors):
            x, y = int(u), int(v)
            if not (0 <= x < W and 0 <= y < H):
                continue

            r, g, b = [int(c) for c in color]
            x0, y0 = x - radius, y - radius
            x1, y1 = x + radius, y + radius

            draw.ellipse(
                [x0, y0, x1, y1],
                fill=(r, g, b, alpha),
            )
    
    def save_time_colorbar(
        self, 
        cmap_name='Reds',   # trajectory에 쓴 colormap 이름
        t_min=0.0,
        t_max=3.0,
        out_path='time_colorbar.png',
        horizontal=True,
        ):
        """
        trajectory에서 사용한 colormap과 시간 범위를 그대로 써서
        별도의 colorbar 이미지를 저장.
        """
        cmap = plt.get_cmap(cmap_name)
        norm = matplotlib.colors.Normalize(vmin=t_min, vmax=t_max)

        # 가로 bar일 때는 figure를 납작하게, 세로면 세로로 길게
        if horizontal:
            fig, ax = plt.subplots(figsize=(4, 0.6))  # 가로 x 세로 (인치)
            orientation = 'horizontal'
        else:
            fig, ax = plt.subplots(figsize=(0.6, 4))
            orientation = 'vertical'

        # 화면에는 아무 것도 안 그려두고, colorbar만 생성
        fig.subplots_adjust(bottom=0.5 if horizontal else 0.05,
                            top=0.95, left=0.1, right=0.9)

        cb = matplotlib.colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            orientation=orientation,
        )

        cb.set_label('time (s)')   # 라벨 텍스트

        # tick 설정 (0, 1, 2, 3 이런 식으로)
        ticks = [t_min, (t_min + t_max)/2, t_max]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.1f}" for t in ticks])

        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved colorbar to {out_path}")
