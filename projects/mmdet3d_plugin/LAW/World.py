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

DATASET_SOURCE={
    0: 'nuScenes',
    1: 'Waymo',
    2: 'Navsim',
}

@DETECTORS.register_module()
class World(VAD_waymo):
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
                **kwargs,
                 ):
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

             # gt: [B,T,2]
            gt = gt_ego_fut_trajs
            if gt.dim() == 4 and gt.size(1) == 1:
                gt = gt[:, 0]
            elif gt.dim() != 3:
                gt = gt.view(bs, -1, 2)

            # mask: [B,T,1], float32, [0,1] clamp
            m = gt_ego_fut_masks
            # [B,T] → [B,T,1]
            if m.dim() == 1:
                m = m.unsqueeze(0)
            if m.dim() == 2:
                m = m.unsqueeze(-1)
            elif m.dim() > 3:
                m = m.view(bs, -1, 1)
            m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            
            # import pdb; pdb.set_trace()
            ego_fut_trajs_w_mask = torch.cat([gt, m], dim=-1)
            pred_ego_fut_trajs, cur_img_feat, pred_img_feat = self.pts_bbox_head(img_feats, img_metas, ego_fut_trajs_w_mask=ego_fut_trajs_w_mask)

            if not is_test:
                # ------- (1) 배치 GT/MASK 수집 -------
                # ------- (2) 모양/타입 표준화 (loss_waypoint와 동일) -------
                preds = pred_ego_fut_trajs.to(torch.float32)

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
        # import pdb;pdb.set_trace()
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

        prev_pred_img_feat = pred_img_feat
        # import pdb; pdb.set_trace()
        
        preds_ego_future_traj, cur_img_feat, pred_img_feat = self.pts_bbox_head(img_feats, img_metas, ego_info)

        loss_rec = self.pts_bbox_head.loss_reconstruction(prev_pred_img_feat, cur_img_feat.detach())
        losses = {'loss_rec': loss_rec * self.wm_loss_weight}

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
        **kwargs
    ):
        
        # if ego_fut_cmd[0][0][0][0][2] != 1:
            # breakpoint()
        # breakpoint()
        # img 위에, ego_fut_trajs (front cam)랑 ego_his_trajs (rear cam에)를 둘 다 viz 해주셈
        # 
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

        
        
        debug=False # DEFAULT: False
        metric_dict = self.simple_test_pts(
            cur_img_feats,
            cur_img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            fut_valid_flag=fut_valid_flag,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
            debug=debug
        )
        if debug:
            ego_fut_preds = metric_dict
            img_metas = cur_img_metas
            pass  # debug breakpoint removed for release
        
        for result_dict in bbox_list:
            result_dict['metric_results'] = metric_dict

        return bbox_list
    
    def simple_test_pts(
        self,
        img_feats,
        img_metas,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        fut_valid_flag=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        debug=False
    ):
        """Test function"""
        B = ego_his_trajs.size(0)
        ego_his_trajs_ = ego_his_trajs.reshape(B, -1)
        ego_lcf_feat_ = ego_lcf_feat.reshape(B, -1)
        ego_fut_cmd_ = ego_fut_cmd.reshape(B, -1)
        ego_info = torch.cat([ego_his_trajs_, ego_lcf_feat_, ego_fut_cmd_], dim=1)

        preds_ego_future_traj, _, _ = self.pts_bbox_head(
                                        img_feats, 
                                        img_metas, 
                                    )
        if debug:
            return preds_ego_future_traj
        with torch.no_grad():
            # pre-process
            if gt_bboxes_3d is not None:
                gt_bbox = gt_bboxes_3d[0][0]
                gt_label = gt_labels_3d[0][0].to('cpu')
                gt_attr_label = gt_attr_labels[0][0].to('cpu').unsqueeze(0)
            else:
                gt_bbox, gt_label, gt_attr_label = None, None, None
            if fut_valid_flag is not None:
                fut_valid_flag = bool(fut_valid_flag[0][0])
            else:
                fut_valid_flag = True

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = preds_ego_future_traj[0]
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            
            ego_fut_preds = ego_fut_preds.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            
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
            if self.call_count % 500 == 0:
                self.compute_and_print_metrics_average()

        return metric_dict_planner_stp3
    
    def compute_and_print_metrics_average(self):
        # compute avg
        avg_metrics = {key: sum(m[key] for m in self.metrics_history) / len(self.metrics_history) for key in self.metrics_history[0]}
        # print avg
        print(f"\n Average metrics after {len(self.metrics_history)} calls: {avg_metrics}")


