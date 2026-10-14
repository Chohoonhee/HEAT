import torch, random
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from torch.nn import functional as F
import math

from mmdet3d.models import builder
from mmdet3d.models.builder import build_loss
from mmdet3d.core import AssignResult, PseudoSampler
from mmdet.core import build_bbox_coder, build_assigner, multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid

from projects.mmdet3d_plugin.LAW.utils import prj_pts_to_img

from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time, json

from projects.mmdet3d_plugin.LAW.dense_heads.utils import get_locations
from projects.mmdet3d_plugin.LAW.utils.visualization import prj_ego_traj_to_2d
# from thop import profile
import pickle


@HEADS.register_module()
class WaypointHead_hist_proto_contrast(BaseModule):
    def __init__(self,
                num_proposals=6,
                #MHA
                hidden_channel=256,
                dim_feedforward=1024,
                num_heads=8,
                dropout=0.0,
                #pos embedding
                depth_step=0.8,
                depth_num=64,
                depth_start = 0,
                position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                stride=32,
                num_views=6,
                #others
                train_cfg=None,
                test_cfg=None,
                use_wm=True,
                num_spatial_token=36,
                num_tf_layers=2,
                num_traj_modal=1,
                trajs_output_dim=2,
                ego_his_timestep=2,
                prototype_path = None,
                action_prototype_path = None,
                mask_blank_view_ctx = False,
                **kwargs,
                ):
        """
        use to predict the waypoints
        """
        super().__init__(**kwargs)
        self.use_wm = use_wm

        # query feature
        self.num_views = num_views
        self.num_proposals = num_proposals
        self.view_query_feat = nn.Parameter(torch.randn(1, self.num_views, hidden_channel, self.num_proposals))
        self.waypoint_query_feat = nn.Parameter(torch.randn(1, self.num_proposals, hidden_channel))
        self.trajs_output_dim = trajs_output_dim
        
        self.prototype_path = prototype_path
        
        self.action_prototype_path = action_prototype_path
        
        
        
        # spatial attn
        spatial_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        ) 
        
        self._spatial_decoder = nn.ModuleList( [
            nn.TransformerDecoder(spatial_decoder_layer, 1) 
            for _ in range(self.num_views)])

        hist_decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_channel,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        self.hist_attn = nn.TransformerDecoder(hist_decoder_layer, 1)

        # wp_attn
        wp_decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_channel,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        self.wp_attn = nn.TransformerDecoder(wp_decoder_layer, 1) # input: Bz, num_token, d_model

        self.wp_attn_refine = nn.TransformerDecoder(wp_decoder_layer, 1) # input: Bz, num_token, d_model

        # world model
        wm_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, num_tf_layers) 

        self.action_aware_encoder = nn.Sequential(
            nn.Linear(hidden_channel + num_proposals*self.trajs_output_dim, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )

        self.action_aware_encoder_init = nn.Sequential(
            nn.Linear(hidden_channel + num_proposals*self.trajs_output_dim, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )

        self.hist_aware_encoder = nn.Sequential(
            nn.Linear(ego_his_timestep*self.trajs_output_dim+5+4, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )

        # self.ego_lcf_encoder = nn.Sequential(
        #     nn.Linear(5, hidden_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_channel, hidden_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_channel, hidden_channel)
        # )

        
        # self.action_aware_encoder = nn.Sequential(
        #     nn.Linear(hidden_channel + 6*2, hidden_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_channel, hidden_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_channel, hidden_channel)
        # )
        
        # loss
        self.loss_plan_reg = build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.loss_plan_rec = nn.MSELoss()

        # head
        self.num_traj_modal = num_traj_modal
        # self.waypoint_head = nn.Sequential(
        #         nn.Linear(hidden_channel, hidden_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(hidden_channel, hidden_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(hidden_channel, self.num_traj_modal* 2)
        #     )
        self.waypoint_head = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, self.num_traj_modal* self.trajs_output_dim)
            )

        self.waypoint_head_init = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, self.num_traj_modal* self.trajs_output_dim)
            )

        # position embedding
        ##img pos embed
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.depth_start = depth_start
        self.stride = stride

        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, hidden_channel*4),
                nn.ReLU(),
                nn.Linear(hidden_channel*4, hidden_channel),
            )

        self.pc_range = nn.Parameter(torch.tensor(point_cloud_range), requires_grad=False)
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        # LID depth
        index = torch.arange(start=0, end=self.depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)
        
        self.temperature = 0.5
        # when True, action_attention injects no prototype context at views whose
        # image is the zero_image.jpg filler (nuScenes SIDE_RIGHT/SIDE_LEFT).
        self.mask_blank_view_ctx = mask_blank_view_ctx

    

        self.action_desc_proj = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel, bias=False),
            nn.ReLU(inplace=True)
        )


        
        self.register_buffer("prototypes", None, persistent=False)
        self.set_prototypes()

    @torch.no_grad()
    def set_prototypes(self):
        """
        P: [V, N, H, W] 텐서. 모델 내 buffer로 등록.
        """
        with open(self.prototype_path, 'rb') as f:
	        self.prototype = pickle.load(f)
        
        with open(self.action_prototype_path, 'rb') as f:
	        self.action_prototype = pickle.load(f)


        prototype_tensor, idx2key, key2idx = self.compact_stack_with_map(self.prototype)

        num_proto, num_view, num_tokens, num_channel = prototype_tensor.shape
        prototype_tensor = prototype_tensor.reshape(num_proto, -1, num_channel)

        self.prototype_tensor = prototype_tensor.detach()
        self.idx2key, self.key2idx = idx2key, key2idx

        self.action_prototype_tensor = self.compact_stack(self.action_prototype).detach()
        

    
    def compact_stack(self, feat_dict):
        """
        feat_dict: {int -> Tensor}
        - 각 값은 (H, W, C) 또는 (T, C) 형태.
        - 예) (8, 8, 256) -> 출력 (N, 8, 8, 256)
            (64, 256)    -> 출력 (N, 64, 256)

        반환:
        rank 3 입력이면 (N, H, W, C)
        rank 2 입력이면 (N, T, C)
        """
        if not feat_dict:
            raise ValueError("feat_dict가 비어 있습니다.")

        keys = sorted(feat_dict.keys())
        tensors = []
        ref_shape = None
        ref_rank = None
        ref_dtype = None
        ref_device = None

        for k in keys:
            t = feat_dict[k]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t)

            if ref_shape is None:
                ref_shape = tuple(t.shape)
                ref_rank = t.dim()
                ref_dtype = t.dtype
                ref_device = t.device

                if ref_rank not in (2, 3):
                    raise ValueError(f"지원하지 않는 텐서 랭크입니다: key {k} -> shape {t.shape}")

                # 필요시 엄격 검증을 켜세요:
                # if ref_rank == 3 and ref_shape != (8, 8, 256):
                #     raise ValueError(f"(H,W,C) 예상 (8,8,256)과 다릅니다: key {k} -> {ref_shape}")
                # if ref_rank == 2 and ref_shape != (64, 256):
                #     raise ValueError(f"(T,C) 예상 (64,256)과 다릅니다: key {k} -> {ref_shape}")
            else:
                if t.dim() != ref_rank:
                    raise ValueError(f"랭크 불일치: key {k} -> rank {t.dim()}, expected {ref_rank}")
                if tuple(t.shape) != ref_shape:
                    raise ValueError(f"shape 불일치: key {k} -> {tuple(t.shape)}, expected {ref_shape}")

                if t.dtype != ref_dtype or t.device != ref_device:
                    t = t.to(device=ref_device, dtype=ref_dtype)

            tensors.append(t)

        out = torch.stack(tensors, dim=0).contiguous()  # rank 3 -> (N,H,W,C), rank 2 -> (N,T,C)
        return out

    def compact_stack_with_map(self, feat_dict, sort_keys=False, device=None, dtype=None, make_contiguous=True):
        """
        feat_dict: {key -> Tensor-like}, 각 값은 rank 2 (T,C) 또는 rank 3 (H,W,C)
        반환:
        out      : (N, T, C) 또는 (N, H, W, C)
        idx2key  : {int_index -> original_key}
        key2idx  : {original_key -> int_index}
        """
        if not feat_dict:
            raise ValueError("feat_dict가 비어 있습니다.")

        # 키 순서 결정: 기본은 삽입 순서, 정렬 원하면 sort_keys=True
        keys = sorted(feat_dict.keys()) if sort_keys else list(feat_dict.keys())

        tensors = []
        ref_shape = None
        ref_rank  = None
        ref_dtype = None
        ref_device = None

        for i, k in enumerate(keys):
            t = feat_dict[k]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t)

            if i == 0:
                if t.dim() not in (2, 3):
                    raise ValueError(f"지원하지 않는 텐서 랭크: key={k}, shape={tuple(t.shape)}")

                ref_shape  = tuple(t.shape)
                ref_rank   = t.dim()
                ref_dtype  = t.dtype
                ref_device = t.device

                target_device = ref_device if device is None else device
                target_dtype  = ref_dtype  if dtype  is None else dtype
            else:
                if t.dim() != ref_rank:
                    raise ValueError(f"랭크 불일치: key={k}, rank={t.dim()}, expected={ref_rank}")
                if tuple(t.shape) != ref_shape:
                    raise ValueError(f"shape 불일치: key={k}, shape={tuple(t.shape)}, expected={ref_shape}")

            # 공통 device/dtype으로 이동
            t = t.to(device=(ref_device if device is None else device),
                    dtype=(ref_dtype  if dtype  is None else dtype))
            if make_contiguous:
                t = t.contiguous()
            tensors.append(t)

        out = torch.stack(tensors, dim=0).contiguous()

        # 인덱스-키 매핑 제공
        idx2key = {i: k for i, k in enumerate(keys)}
        key2idx = {k: i for i, k in enumerate(keys)}
        return out, idx2key, key2idx


    def prepare_location(self, img_metas, img_feats):
        # breakpoint()
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = img_feats.shape[:2]
        x = img_feats.flatten(0, 1)
        location = get_locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location
    
    def img_position_embeding(self, img_feats, img_metas):
        """
        from streampetr
        """
        eps = 1e-5
        B, num_views, C, H, W = img_feats.shape
        assert num_views == self.num_views, 'num_views should be equal to self.num_views'
        BN = B * num_views
        num_sample_tokens = num_views * H * W
        LEN = num_sample_tokens
        img_pixel_locations = self.prepare_location(img_metas, img_feats)

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        img_pixel_locations[..., 0] = img_pixel_locations[..., 0] * pad_w
        img_pixel_locations[..., 1] = img_pixel_locations[..., 1] * pad_h

        # Depth
        D = self.coords_d.shape[0]
        pixel_centers = img_pixel_locations.detach().view(B, LEN, 1, 2).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([pixel_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        # batch>1: per-sample camera calibration (was sample-0's applied to the whole batch)
        lidar2img = np.stack([np.stack(m['lidar2img'])[:num_views] for m in img_metas])
        lidar2img = torch.from_numpy(lidar2img).to(img_feats.device).float()  # [B, V, 4, 4]
        img2lidars = lidar2img.inverse()
        img2lidars = img2lidars.view(B, num_views, 1, 1, 4, 4).repeat(1, 1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3]) #normalize
        coords3d = coords3d.reshape(B, -1, D*3)
        coords3d = coords3d.clamp_(min=1e-4, max=1-1e-4)  # ★ 추가

        pos_embed  = inverse_sigmoid(coords3d) #(B, num_views*H*W, 3*64)
        coords_position_embeding = self.position_encoder(pos_embed)
        return coords_position_embeding
    
    def forward(self, img_feat, img_metas, ego_info=None, is_test=False):
        # init
        losses = {}
        Bz, num_views, num_channels, height, width = img_feat.shape
        init_view_query_feat = self.view_query_feat.clone().repeat(Bz, 1, 1, 1).permute(0, 1, 3, 2)
        init_waypoint_query_feat = self.waypoint_query_feat.clone().repeat(Bz, 1, 1)

        # img pos emb
        img_pos = self.img_position_embeding(img_feat, img_metas)
        img_pos = img_pos.reshape(Bz, num_views, height, width, num_channels)
        img_pos = img_pos.permute(0, 1, 4, 2, 3)
        img_feat_emb = img_feat + img_pos   

        # spatial view feat
        img_feat_emb = img_feat_emb.reshape(Bz, num_views, num_channels, height*width).permute(0, 1, 3, 2)
        spatial_view_feat = torch.zeros_like(init_view_query_feat)
        for i in range(self.num_views):
            spatial_view_feat[:, i] = self._spatial_decoder[i](init_view_query_feat[:, i], img_feat_emb[:, i])

        batch_size, num_view, num_tokens, num_channel = spatial_view_feat.shape
        spatial_view_feat = spatial_view_feat.reshape(batch_size, -1, num_channel)

        # tokens whose view has no real observation (nuScenes blank fill)
        blank_tok_mask = self._blank_view_token_mask(
            img_metas, num_view, num_tokens, spatial_view_feat.device)
        ctx_mask = blank_tok_mask if self.mask_blank_view_ctx else None


        ################## our module contrastive ##################
        # batch>1: one cluster index PER SAMPLE (was a single scalar for the whole batch)
        if 'cluster' in img_metas[0].keys():
            _cl = []
            for m in img_metas:
                v = m['cluster']
                v = v.data if hasattr(v, 'data') else v
                _cl.append(self.key2idx[int(torch.as_tensor(v).reshape(-1)[0].item())])
            cluster_index = torch.as_tensor(_cl, dtype=torch.long, device=spatial_view_feat.device)
        else:
            cluster_index = torch.zeros(batch_size, dtype=torch.long,
                                        device=spatial_view_feat.device)

        if not is_test:
            # nuScenes has no SIDE_RIGHT/SIDE_LEFT: views 6,7 are filled with
            # samples/zero_image.jpg. Those tokens carry no observation, and the
            # prototype bank is built with them excluded, so drop them from the loss.
            # blank_tok_mask above already encodes this (None => nothing blank)
            valid_tok = blank_tok_mask
            loss_contrast = self.contrastive_proto_loss(
                                feat=spatial_view_feat,        # [B,64,256]
                                cluster_idx=cluster_index,
                                tau=0.07,
                                detach_proto=True,
                                valid_mask=valid_tok,
                            )
        else:
            loss_contrast = None

        # history_traj update
        
        # batch>1: stack each sample's own ego state (was sample-0 reshaped to the batch,
        # which crashes at B>1 because one sample's tensor cannot fill B rows)
        def _stack_meta(key):
            rows = []
            for m in img_metas:
                v = m[key]
                if not isinstance(v, torch.Tensor):
                    v = v.data if hasattr(v, 'data') else v
                rows.append(torch.as_tensor(v).reshape(-1))
            return torch.stack(rows, dim=0).unsqueeze(1).repeat(1, num_tokens, 1)

        history_traj = _stack_meta('ego_his_trajs')
        ego_lcf_feat = _stack_meta('ego_lcf_feat')
        # ego_lcf_feat = self.ego_lcf_encoder(ego_lcf_feat.to(img_feat.device))

        ego_fut_cmd = _stack_meta('ego_fut_cmd')
        
        
        
        # if os.environ['DEBUG'] =='JY': import pdb; pdb.set_trace()
        # history_traj_feat = self.hist_aware_encoder(torch.cat([history_traj, ego_lcf_feat], dim=-1).to(img_feat.device).float())
        history_traj_feat = self.hist_aware_encoder(torch.cat([history_traj, ego_lcf_feat, ego_fut_cmd], dim=-1).to(img_feat.device).float())

        ##########################################################



        # lcf_feat (ego_state) update



        ################### our module attention #######################
        # predict wp
        init_waypoint_query_feat = self.hist_attn(init_waypoint_query_feat, history_traj_feat) 
        updated_waypoint_query_feat = self.wp_attn(init_waypoint_query_feat, spatial_view_feat) #final_view_feat.shape torch.Size([1, 1440, 256])
        init_waypoint = self.waypoint_head_init(updated_waypoint_query_feat)
        
        if self.num_traj_modal > 1:
            bz, traj_len, _ = init_waypoint.shape
            init_waypoint = init_waypoint.view(bz, traj_len, self.num_traj_modal, self.trajs_output_dim)  # NOTE: 기원 250913 수정
            # batch>1: pick each sample's mode from its OWN command (was sample-0's for all)
            _idx = []
            for b in range(bz):
                cmd = img_metas[b]['ego_fut_cmd']
                if isinstance(cmd, torch.Tensor):
                    cmd = cmd.to(init_waypoint.device, dtype=torch.float32)
                    ci = int(torch.argmax(cmd, dim=-1).view(-1)[0].item()) if cmd.numel()>1 else int(cmd.view(-1)[0].item())
                else:
                    cmd_np = np.asarray(cmd, dtype=np.float32)
                    ci = int(np.argmax(cmd_np)) if cmd_np.size>1 else int(cmd_np.flat[0])
                _idx.append(max(0, min(self.num_traj_modal-1, ci)))
            cmd_idx = torch.as_tensor(_idx, device=init_waypoint.device)
            batch_idx = torch.arange(bz, device=init_waypoint.device)
            init_waypoint = init_waypoint[batch_idx, :, cmd_idx]
        

        wp_pre_reshape = init_waypoint 
        
        init_waypoint = init_waypoint.reshape(batch_size, 1, -1).repeat(1, num_view * num_tokens, 1)
        init_view_query_feat_with_ego = torch.cat([spatial_view_feat, init_waypoint], dim=-1) 
        
        
        updated_waypoint_query_feat = self.action_aware_encoder_init(init_view_query_feat_with_ego)
        updated_waypoint_query_feat = self.action_attention(updated_waypoint_query_feat, valid_mask=ctx_mask)
        updated_waypoint_query_feat = self.wp_attn_refine(init_waypoint_query_feat, updated_waypoint_query_feat)
        ################################################################

        cur_waypoint = self.waypoint_head(updated_waypoint_query_feat)




        # if self.num_traj_modal > 1:
        #     # assert self.num_traj_modal == 3
        #     bz, traj_len, _ = cur_waypoint.shape
        #     # cur_waypoint = cur_waypoint.reshape(bz, traj_len, self.num_traj_modal, 2)
        #     cur_waypoint = cur_waypoint.reshape(bz, traj_len, self.num_traj_modal, self.trajs_output_dim)
        #     ego_cmd = img_metas[0]['ego_fut_cmd'].to(img_feat.device)[0, 0]
        #     cur_waypoint = cur_waypoint[: ,: ,ego_cmd == 1].squeeze(2) # cmd에 맞는 traj 선택.
        if self.num_traj_modal > 1:
            bz, traj_len, _ = cur_waypoint.shape
            # cur_waypoint = cur_waypoint.view(bz, traj_len, self.num_traj_modal, 2)
            cur_waypoint = cur_waypoint.view(bz, traj_len, self.num_traj_modal, self.trajs_output_dim)  # NOTE: 기원 250913 수정
            # reuses cmd_idx / batch_idx from the init head above (same num_traj_modal>1 branch)
            cur_waypoint = cur_waypoint[batch_idx, :, cmd_idx]



        # world model prediction
        # use_wm=False disables the future-latent branch entirely, so
        # the caller also drops L_recon and the planner never sees a world model.
        # breakpoint()
        wm_next_latent = self.wm_prediction(spatial_view_feat, cur_waypoint) \
            if self.use_wm else None

        return cur_waypoint, spatial_view_feat, wm_next_latent, wp_pre_reshape, loss_contrast

    
    def contrastive_proto_loss(self,
        feat,            # [B, T, C]
        cluster_idx,     # int (legacy) or LongTensor [B]
        tau: float = 0.07,
        detach_proto: bool = False,
        valid_mask=None, # optional BoolTensor [B, T]; False = token carries no observation
    ):
        """
        마스크 없이 타임스텝별 InfoNCE (cosine + temperature).
        반환: 스칼라 loss
        """
        # --- device/dtype 정렬 ---
        target_device = feat.device
        calc_dtype = torch.float32

        # proto 불러오고 detach 옵션 적용 + device/dtype 이동
        p_raw = self.prototype_tensor.detach() if detach_proto else self.prototype_tensor
        p = p_raw.to(device=target_device, dtype=calc_dtype).contiguous()  # [N, T, C]

        # feat도 동일하게 정렬 (batch>1: keep the batch dim instead of taking feat[0])
        f = feat.to(device=target_device, dtype=calc_dtype).contiguous()  # [B, T, C]

        # shape 검증 (정렬 후)
        assert feat.dim() == 3 and p.dim() == 3, "feat: [B,T,C], proto: [N,T,C]"
        B, T, C = feat.shape
        N, Tp, Cp = p.shape

        assert T == Tp and C == Cp, "shape mismatch"

        # cluster_idx: int (legacy, one label for the batch) or LongTensor [B] (per sample)
        if not torch.is_tensor(cluster_idx):
            cluster_idx = torch.full((B,), int(cluster_idx), dtype=torch.long,
                                     device=target_device)
        cluster_idx = cluster_idx.to(device=target_device, dtype=torch.long).reshape(-1)
        assert cluster_idx.numel() == B, "cluster_idx must have one entry per sample"
        assert bool(((cluster_idx >= 0) & (cluster_idx < N)).all()), "cluster_idx out of range"

        # tau도 같은 device/dtype 텐서로
        tau_t = torch.as_tensor(tau, device=target_device, dtype=calc_dtype)

        # --- cosine normalize ---
        f = F.normalize(f, dim=-1)     # [B,T,C]
        p = F.normalize(p, dim=-1)     # [N,T,C]

        # --- logits: [B, T, N] ---
        logits = torch.einsum("ntc,btc->btn", p, f) / tau_t

        # --- targets & loss (mean over B*T, identical to the old mean over T at B=1) ---
        targets = cluster_idx.view(B, 1).expand(B, T)
        ce = F.cross_entropy(logits.reshape(B * T, N), targets.reshape(B * T),
                             reduction="none").view(B, T)
        if valid_mask is None:
            loss = ce.mean()
        else:
            vm = valid_mask.to(device=ce.device, dtype=ce.dtype)
            loss = (ce * vm).sum() / vm.sum().clamp(min=1.0)

        return loss

    def loss_reconstruction(self, 
            reconstructed_view_query_feat,
            observed_view_query_feat,
            ):
        loss_rec = self.loss_plan_rec(reconstructed_view_query_feat, observed_view_query_feat)
        return loss_rec

    
    def _blank_view_token_mask(self, img_metas, num_view, num_tokens, device):
        """[B, num_view*num_tokens] bool mask; False where a view's image is the
        zero_image.jpg filler (nuScenes has no SIDE_RIGHT/SIDE_LEFT).
        Returns None when every view of every sample carries a real image."""
        view_ok = []
        for m in img_metas:
            fn = m.get('filename', None) if hasattr(m, 'get') else None
            if fn is None:
                view_ok.append([True] * num_view)
            else:
                view_ok.append([('zero_image' not in str(f)) for f in fn[:num_view]])
        valid_view = torch.as_tensor(view_ok, dtype=torch.bool, device=device)
        if bool(valid_view.all()):
            return None
        return valid_view.repeat_interleave(num_tokens, dim=1)

    def action_attention(self, x, valid_mask=None):
        assert self.action_prototype_tensor is not None, "call set_prototypes(P[V,T,C]) first"
        B, T, C = x.shape
        # import pdb; pdb.set_trace()
        V, TP, CP = self.action_prototype_tensor.shape
        assert T == TP and C == CP, f"shape mismatch: x {x.shape}, P {self.action_prototype_tensor.shape}"

        # P = self.action_prototype_tensor  # [V,T,C]
        P = self.action_prototype_tensor.to(x.device, dtype=x.dtype)
        # 1) desc_proj 임베딩 후 정규화 (코사인 유사도용)
        x_desc = self.action_desc_proj(x)                  # [B,T,H]
        x_desc = F.normalize(x_desc, dim=-1)

        P_desc = self.action_desc_proj(P)                  # [V,T,H] (buffer → 투영만 학습)
        P_desc = F.normalize(P_desc, dim=-1)

        # 2) 같은 토큰 인덱스끼리 유사도 → 어텐션
        # sim[b,t,v] = <x_desc[b,t], P_desc[v,t]>
        sim = torch.einsum('bth,vth->btv', x_desc, P_desc)    # [B,T,V]
        attn = torch.softmax(sim / self.temperature, dim=-1)  # [B,T,V]
        
        # 3) 컨텍스트: 원본 P 값으로 가중합 (value=원본)
        # ctx[b,t,c] = Σ_v attn[b,t,v] * P[v,t,c]
        ctx = torch.einsum('btv,vtc->btc', attn, P)           # [B,T,C]

        # 4) Residual 합성
        if valid_mask is not None:
            # no observation at these tokens -> do not inject prototype content
            ctx = ctx * valid_mask.to(ctx.dtype).unsqueeze(-1)
        out = x + ctx                                         # [B,T,C]


        return out
    
    def wm_prediction(self, view_query_feat, cur_waypoint):
        batch_size, num_tokens, num_channel = view_query_feat.shape
        cur_waypoint = cur_waypoint.reshape(batch_size, 1, -1).repeat(1, num_tokens, 1)
        cur_view_query_feat_with_ego = torch.cat([view_query_feat, cur_waypoint], dim=-1) 
        
        
        action_aware_latent = self.action_aware_encoder(cur_view_query_feat_with_ego)

        

        wm_next_latent = self._wm_decoder(action_aware_latent, action_aware_latent)
        
        return wm_next_latent
    
    def loss_3d(self, 
            preds_ego_future_traj,
            gt_ego_future_traj,
            gt_ego_future_traj_mask,
            ego_info=None,
            ):
        loss_waypoint = self.loss_plan_reg(preds_ego_future_traj, gt_ego_future_traj, gt_ego_future_traj_mask)
        # print("pred:", preds_ego_future_traj)
        # print("gt:",gt_ego_future_traj)
        return loss_waypoint
