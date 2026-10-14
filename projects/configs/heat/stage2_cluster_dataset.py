_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = len(class_names)

# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

# Stage-2 outputs (proto/noimg_make_proto.py): visual prototypes and action memory
action_prototype_path = './proto/18_cluster/all/action_aware_latent_prototype_vectors.pkl'
prototype_path = './proto/18_cluster/all/img_feat_prototype_vectors.pkl'


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 100
queue_length = 4 
total_epochs = 48

with_heading = True
if with_heading==True:
    trajs_output_dim = 3
else: 
    trajs_output_dim = 2
heading_mask = False


absolute_path = True     # 절대 경로로 사용
if absolute_path==True:
    ego_his_timestep = 3
else:
    ego_his_timestep = 2

is_absolute_path = True   


model = dict(
    type='LAW_proto_contrast',
    use_grid_mask=True,
    video_test_mode=True,
    use_multi_view=True,
    use_swin=True,
    swin_input_channel=768,
    hidden_channel=256,
    prototype_path = prototype_path,
    img_backbone=dict(
        type='SwinTransformer3D', # We use only a single frame, which makes it the same as Swin-tiny.
        arch='tiny',
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth',
        pretrained2d=True,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.2,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True
    ),
    pts_bbox_head=dict(
        type='WaypointHead_hist_proto_contrast',
        num_proposals=8,
        num_views=8,
        hidden_channel=256,
        num_heads=8,
        dropout=0.1,
        use_wm=True,
        num_traj_modal=4,
        trajs_output_dim=trajs_output_dim,
        prototype_path = prototype_path,
        ego_his_timestep=ego_his_timestep,
        action_prototype_path = action_prototype_path,
        mask_blank_view_ctx=True,  # exclude nuScenes padded views from memory retrieval
        ),
    )


dataset_type = 'VADCustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline_nus = [
    # dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='RandomScaleImageMultiViewImage', scales=[0.48]),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',\
         keys=[ # 'img',
               'gt_bboxes_3d', 'gt_labels_3d', 'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'])
]


dataset_type_waymo = 'CustomWaymoDataset_new_subsampling_251014'
data_root_waymo = 'data/waymo_e2e_dataset/'

train_pipeline_waymo = [
    # dict(type='LoadMultiViewImageFromFilesWaymo', to_float32=True),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_attr_label=False),
    # dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='CustomObjectNameFilter', classes=class_names),
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='RandomScaleImageMultiViewImage', scales=[0.65]),
    # dict(type='RandomScaleImageMultiViewImage', scales=[0.78]),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D',\
         keys=[ #'img',
               'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat'])
]

dataset_type_navsim = 'NavsimDataset_GW'

# NOTE: navsim dataset path (9_4k_8ts_train, 6k_6ts_val)

# NOTE: navsim dataset path (28k_6ts_train, 6k_6ts_val)

# NOTE: navsim dataset full path (full_train, 6k_6ts_val)
train_data_dir_nav='data/navsim/cache/train'
val_test_data_dir_nav='data/navsim/cache/val'

data_root_navsim = 'data/navsim/openscene'

file_client_args = dict(backend='disk')
train_pipeline_navsim = [
    # dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    # dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='CustomObjectNameFilter', classes=class_names),
    # dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_ego=True),
    dict(type='CustomCollect3D_navsim',\
         keys=[ #'img', 
               'ego_his_trajs',
               'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5,
         file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_label=False, with_ego=True),
            dict(type='CustomCollect3D',\
                 keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                       'ego_lcf_feat', 'gt_attr_labels'])])
]

nusc_cluster_data_dir = None
waymo_cluster_data_dir = None
navsim_cluster_data_dir = None






data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    # workers_per_gpu=0,
    train_nusc=dict(
        type=dataset_type,
        data_root=data_root,
        interval_2frames=True,
        # ann_file=data_root + 'vad_nuscenes_heading_infos_temporal_train_9_4k.pkl',
        ann_file=data_root + 'vad_nuscenes_heading_infos_temporal_train.pkl', # for 28.1 k
        pipeline=train_pipeline_nus,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        map_classes=map_classes,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        box_type_3d='LiDAR',
        custom_eval_version='vad_nusc_detection_cvpr_2019',
        with_heading=with_heading,
        # heading_mask=heading_mask,
        absolute_path = absolute_path,
        cluster_data_dir = nusc_cluster_data_dir,
        skip_cam=True),
    train_waymo=dict(
        type=dataset_type_waymo,
        data_root=data_root_waymo,
        interval_2frames=True,
        ann_file=data_root_waymo + 'vad_nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline_waymo,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        queue_length=queue_length,
        map_classes=map_classes,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        box_type_3d='LiDAR',
        custom_eval_version='vad_nusc_detection_cvpr_2019',
        with_heading=with_heading,
        # heading_mask=heading_mask,
        absolute_path = absolute_path,
        cluster_data_dir = waymo_cluster_data_dir,
        skip_cam=True),
    train_navsim=dict(type=dataset_type_navsim,
             data_root=data_root_navsim,
             pipeline=train_pipeline_navsim,  # TODO: navsim용으로, pipeline을 제거함.
             queue_length=queue_length,
             data_dir=train_data_dir_nav,
             classes=class_names,
             with_heading=with_heading,
             absolute_path = absolute_path,
             cluster_data_dir = navsim_cluster_data_dir,
             skip_cam=True,),           # TODO: navsim의 class에 맞게 수정해야함. 지금은 뉴신스의 class를 그대로 사용함.
    val=dict(type=dataset_type,
             data_root=data_root, 
             queue_length=queue_length,
             pc_range=point_cloud_range,
             interval_2frames=True,
             ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             map_classes=map_classes,
             map_ann_file=data_root + 'nuscenes_map_anns_val.json',
             map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
             map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
             use_pkl_result=True,
             custom_eval_version='vad_nusc_detection_cvpr_2019'),
    test=dict(type=dataset_type,
              data_root=data_root,
              queue_length=queue_length,
              interval_2frames=True,
              pc_range=point_cloud_range,
              ann_file=data_root + 'vad_nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality, samples_per_gpu=1,
              map_classes=map_classes,
              map_ann_file=data_root + 'nuscenes_map_anns_val.json',
              map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
              map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
              use_pkl_result=True,
              custom_eval_version='vad_nusc_detection_cvpr_2019'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=5e-5,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

evaluation = dict(interval=total_epochs + 1, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)


custom_hooks = [dict(type='CustomSetEpochInfoHook')]

