from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import uuid
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.training.dataset import Dataset
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.abstract_agent import AbstractAgent

import random
import torch
import numpy as np

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    Helper function to cache features and targets of learnable agent.
    :param args: arguments for caching
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    agent: AbstractAgent = instantiate(cfg.agent)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    logger.info(f"Extracted {len(scene_loader.tokens)} scenarios for thread_id={thread_id}, node_id={node_id}.")

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        cfg=cfg,
        sampling_percent=cfg.sampling_percent,
    )       # sampling_percent는 내가 수정함.
    # breakpoint()
    return []


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for dataset caching script.
    :param cfg: omegaconf dictionary
    """
    # previous version
    # cfg.sensor_blobs_path = "data/navsim/openscene/sensor_blobs/trainval"
    # cfg.navsim_log_path = "data/navsim/openscene/navsim_logs/trainval"  # << 여기서 강제로 덮어쓰기
    
    # law model version
    cfg.sensor_blobs_path = "data/navsim/openscene/sensor_blobs/trainval"
    cfg.navsim_log_path = "data/navsim/openscene/navsim_logs/trainval"  # << 여기서 강제로 덮어쓰기
    # breakpoint()
    if cfg.sampling_percent == 1.0:
        cfg.cache_path="navsim_cache_out/training_cache_law"
    else:
        cfg.cache_path=f"navsim_cache_out/training_cache_law_subsampled_to_{cfg.sampling_percent*100}_percent"
        
    # breakpoint()
    if cfg.contain_heading==True:
        if cfg.sampling_percent == 1.0:
            cfg.cache_path="navsim_cache_out/training_cache_law_contain_heading"
        else:
            cfg.cache_path=f"navsim_cache_out/training_cache_law_contain_heading_subsampled_to_{cfg.sampling_percent*100}_percent"
            
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    single_worker = bool(os.getenv("SINGLE_WORKER", "0") == "1" or getattr(cfg, "single_worker", False))

    if not single_worker:
        logger.info("Building Worker")
        worker: WorkerPool = instantiate(cfg.worker)

    logger.info("Building SceneLoader")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    logger.info(f"Extracted {len(scene_loader)} scenarios for training/validation dataset")

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    # ---- 여기만 바꿉니다: worker_map 우회 ----
    if single_worker:
        logger.info("[SingleWorker] Running sequentially without worker pool")
        for dp in data_points:
            # cache_features는 List[Dict]를 받으므로 [dp]로 감싸서 호출
            cache_features([dp])
    else:
        _ = worker_map(worker, cache_features, data_points)
        
    logger.info(f"Finished caching {len(scene_loader)} scenarios for training/validation dataset")


if __name__ == "__main__":
    main()
