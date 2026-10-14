from mmcv.utils.registry import Registry, build_from_cfg

SAMPLER = Registry('sampler')

# (Pdb) cfg                                                                                                                                                                                  
# {'type': 'DistributedGroupSampler'}  
# (Pdb) default_args
# {'dataset': <projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset.VADCustomNuScenesDataset object at 0x7f55c615d2b0>, 
# 'samples_per_gpu': 1, 'num_replicas': 1, 'rank': 0, 'seed': 0}
# (Pdb) SAMPLER
# Registry(name=sampler, items={'DistributedGroupSampler': 
# <class 'projects.mmdet3d_plugin.datasets.samplers.group_sampler.DistributedGroupSampler'>, 
# 'InfiniteGroupEachSampleInBatchSampler': <class 'projects.mmdet3d_plugin.datasets.samplers.group_sampler.InfiniteGroupEachSampleInBatchSampler'>,
#  'DistributedSampler': <class 'projects.mmdet3d_plugin.datasets.samplers.distributed_sampler.DistributedSampler'>})

def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, SAMPLER, default_args)
