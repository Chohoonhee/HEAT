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
class LAW_proto_contrast(VAD_waymo):
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
            pred_ego_fut_trajs, cur_img_feat, pred_img_feat, init_pred_ego_fut_trajs, loss_contrast = \
                self.pts_bbox_head(img_feats, img_metas)

            if not is_test:
                # ------- (1) 배치 GT/MASK 수집 -------
                gts, ms = [], []
                for meta in img_metas:
                    gts.append(to32(meta['ego_fut_trajs']))
                    m = to32(meta['ego_fut_masks'])  # [1,T] or [T]
                    while m.dim() >= 2 and m.size(0) == 1:
                        m = m.squeeze(0)             # -> [T]
                    ms.append(m)

                gt_ego_fut_trajs = torch.stack(gts, dim=0)  # [B, ?, 2]
                gt_ego_fut_masks = torch.stack(ms,  dim=0)  # [B, T]

                # ------- (2) 모양/타입 표준화 -------
                preds = pred_ego_fut_trajs.to(torch.float32)

                gt = gt_ego_fut_trajs
                if gt.dim() == 4 and gt.size(1) == 1:
                    gt = gt[:, 0]                    # [B,T,2]
                elif gt.dim() != 3:
                    gt = gt.view(preds.size(0), -1, 2)

                m = gt_ego_fut_masks
                if m.dim() == 1:
                    m = m.unsqueeze(0)               # [T] -> [1,T]
                if m.dim() == 2:
                    m = m.unsqueeze(-1)              # [B,T] -> [B,T,1]
                elif m.dim() > 3:
                    m = m.view(preds.size(0), -1, 1)
                m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

                # 길이 맞추기 (final preds)
                T = min(preds.size(1), gt.size(1), m.size(1))
                preds_f, gt_f, m_f = preds[:, :T], gt[:, :T], m[:, :T]

                # ------- (3) 기존(final) loss + 안전 폴백 -------
                loss_prev_raw = self.pts_bbox_head.loss_3d(preds_f, gt_f, m_f)
                with torch.no_grad():
                    diff = (preds_f - gt_f).abs()
                    wsum = m_f.sum().clamp_(min=1.0)
                    loss_prev_safe = (diff * m_f).sum() / wsum
                loss_prev = loss_prev_safe if ((not torch.isfinite(loss_prev_raw)) or (loss_prev_raw.abs() > 1e6)) \
                                        else loss_prev_raw

                # ------- (4) init 예측에 대한 동일 손실 (있으면) -------
                if init_pred_ego_fut_trajs is not None:
                    init_preds = init_pred_ego_fut_trajs.to(torch.float32)
                    T_i = min(init_preds.size(1), gt.size(1), m.size(1))
                    preds_i, gt_i, m_i = init_preds[:, :T_i], gt[:, :T_i], m[:, :T_i]
                    loss_init_raw = self.pts_bbox_head.loss_3d(preds_i, gt_i, m_i)
                    with torch.no_grad():
                        diff_i = (preds_i - gt_i).abs()
                        wsum_i = m_i.sum().clamp_(min=1.0)
                        loss_init_safe = (diff_i * m_i).sum() / wsum_i
                    loss_init = loss_init_safe if ((not torch.isfinite(loss_init_raw)) or (loss_init_raw.abs() > 1e6)) \
                                            else loss_init_raw
                else:
                    loss_init = torch.tensor(0.0, device=preds.device, dtype=torch.float32)

                # ------- (5) combine: final + lambda_1 * init (lambda_1 = 0.2) -------
                loss_combined = loss_prev + 0.2 * loss_init
                losses[f'prev_frame_loss_waypoint_{i}'] = loss_combined

                if loss_contrast is not None:
                    losses[f'prev_contrast_{i}'] = 0.005 * loss_contrast

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
        preds_ego_future_traj, cur_img_feat, pred_img_feat, init_pred_ego_fut_trajs, loss_contrast = \
            self.pts_bbox_head(img_feats, img_metas, ego_info)

        # reconstruction loss
        loss_rec = self.pts_bbox_head.loss_reconstruction(prev_pred_img_feat, cur_img_feat.detach())
        losses = {'loss_rec': loss_rec * self.wm_loss_weight}

        # ---- 표준화된 GT/Mask ----
        gt = to32(ego_fut_trajs)
        if gt.dim() == 4 and gt.size(1) == 1:
            gt = gt[:, 0]                          # [B,T,2]
        elif gt.dim() != 3:
            gt = gt.view(B, -1, 2)

        m = to32(ego_fut_masks)
        while m.dim() >= 3 and m.size(1) == 1:
            m = m.squeeze(1)
        if m.dim() == 2:
            m = m.unsqueeze(-1)
        elif m.dim() > 3:
            m = m.view(B, -1, 1)
        m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        # ---- helper: 안전 로스 계산 함수 ----
        def safe_loss(preds_in, gt_in, m_in, tag):
            T = min(preds_in.size(1), gt_in.size(1), m_in.size(1))
            p, g, mm = preds_in[:, :T], gt_in[:, :T], m_in[:, :T]

            raw = self.pts_bbox_head.loss_3d(p, g, mm)
            with torch.no_grad():
                diff = (p - g).abs()
                wsum = mm.sum().clamp_(min=1.0)
                safe = (diff * mm).sum() / wsum
                if (not torch.isfinite(raw)) or (raw.abs() > 1e6):
                    sid = img_metas[0].get('sample_idx', 'unk')
                    print(f'[rank{rank}] {tag} loss explode. raw={raw.item():.3e} safe={safe.item():.3e} '
                        f'sid={sid} shapes P{tuple(p.shape)} G{tuple(g.shape)} M{tuple(mm.shape)} '
                        f'Pmin/max=({p.min().item():.3f},{p.max().item():.3f}) '
                        f'Gmin/max=({g.min().item():.3f},{g.max().item():.3f}) Msum={mm.sum().item():.1f}')
            return safe if ((not torch.isfinite(raw)) or (raw.abs() > 1e6)) else raw

        # ---- final(preds) loss ----
        preds = preds_ego_future_traj.to(torch.float32)
        loss_waypoint_final = safe_loss(preds, gt, m, tag='FINAL')

        # ---- init loss (있을 때만) ----
        if init_pred_ego_fut_trajs is not None:
            init_preds = init_pred_ego_fut_trajs.to(torch.float32)
            loss_waypoint_init = safe_loss(init_preds, gt, m, tag='INIT')
        else:
            loss_waypoint_init = torch.tensor(0.0, device=preds.device, dtype=torch.float32)

        # ---- 합성 & 최종 업데이트 ----
        # final + lambda_1 * init (lambda_1 = 0.2, Eq. (15))
        loss_waypoint = loss_waypoint_final + 0.2 * loss_waypoint_init
        losses['loss_waypoint'] = loss_waypoint

        if loss_contrast is not None:
            losses['loss_contrast'] = 0.005 * loss_contrast

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

        preds_ego_future_traj, _, _, _, _ = self.pts_bbox_head(
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
        
        # breakpoint()
        if kwargs.get("navsim_inference", False):
            # breakpoint()
            bbox_results = self.simple_test_navsim(
                img_metas=img_metas,
                img=img,
                is_absolute_path=is_absolute_path,
                **kwargs
            )
        else:
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

        # breakpoint()
        preds_ego_future_traj, _, _, _, _ = self.pts_bbox_head(
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
            
            ego_fut_preds = ego_fut_preds.cumsum(dim=-2)[:, :2]
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)[:, :2]

            # if os.environ['DEBUG'] =='JY': import pdb; pdb.set_trace()
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


