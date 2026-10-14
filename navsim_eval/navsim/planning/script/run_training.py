from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

import zlib
import struct
import gzip  # (선택) BadGzipFile도 잡고 싶으면 사용
from torch.utils.data import Subset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

import pickle

def save_to_pkl(data: Dataset, file_path: Path) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved data to {file_path}")

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    # breakpoint()
    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    # breakpoint()

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )


    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    # breakpoint()
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    # breakpoint()
    agent: AbstractAgent = instantiate(cfg.agent)
    # breakpoint()
    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )
    # breakpoint()
    # cfg.use_cache_without_dataset = False # TODO: 데이터 생성시에는 이걸로 하드코딩 해두었음.
    # cfg.force_cache_computation = True # TODO: 데이터 생성시에는 이걸로 하드코딩 해두었음.
    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        # breakpoint()
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        # breakpoint()
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    
    # breakpoint()        # train data를 잘 맞게 수정하기
    # Save to .pkl files
    # >>> 옵션: 깨진 샘플 스킵 기능 스위치
    skip_corrupted = True  # >>>

    # >>> 저장 & 유효 인덱스 수집
    save_to_pkl_flag = True  # NOTE: 저장용 코드. law에서 navsim 데이터를 사용하기 위해서임.
    if save_to_pkl_flag:
        import pickle
        from pathlib import Path
        from tqdm import tqdm
        if cfg.sampling_percent == 1.0:
            if cfg.contain_heading==True:
                save_dir_train = Path("navsim_cache_out/pkl_extracted_heading/train")
                save_dir_val = Path("navsim_cache_out/pkl_extracted_heading/val")
            else:
                save_dir_train = Path("navsim_cache_out/pkl_extracted/train")
                save_dir_val = Path("navsim_cache_out/pkl_extracted/val")                
        else:
            if cfg.contain_heading==True:
                save_dir_train = Path(f"navsim_cache_out/pkl_extracted_heading_subsampled_{cfg.sampling_percent*100}_percent/train")
                save_dir_val = Path(f"navsim_cache_out/pkl_extracted_heading_subsampled_{cfg.sampling_percent*100}_percent/val")
            else:
                save_dir_train = Path(f"navsim_cache_out/pkl_extracted_subsampled_{cfg.sampling_percent*100}_percent/train")
                save_dir_val = Path(f"navsim_cache_out/pkl_extracted_subsampled_{cfg.sampling_percent*100}_percent/val")        
        save_dir_train.mkdir(parents=True, exist_ok=True)
        save_dir_val.mkdir(parents=True, exist_ok=True)

        # >>> 유효/스킵 인덱스 추적 리스트
        valid_train_indices = []  # >>>
        skipped_train_indices = []  # >>>
        valid_val_indices = []  # >>>
        skipped_val_indices = []  # >>>

        # >>> 안전한 로드에 사용할 예외 튜플
        load_errors = (EOFError, zlib.error, pickle.UnpicklingError, OSError, struct.error)  # >>>

        # Train 데이터 저장(깨진 항목 skip)
        for i in tqdm(range(len(train_data)), desc="Saving train samples"):
            try:
                sample_train = train_data[i]   # (feature_dict, target_dict)
            except load_errors as e:
                skipped_train_indices.append(i)
                # 토큰/경로 힌트 로그(가능하면)
                token = None
                try:
                    token = getattr(train_data, "tokens", None)
                    token = token[i] if token is not None and i < len(token) else None
                except Exception:
                    pass
                logger.warning(f"[train] Skip corrupted sample idx={i}, token={token}, err={repr(e)}")
                continue
            except Exception as e:
                # 알 수 없는 에러도 스킵(원하면 raise로 바꿔도 됨)
                skipped_train_indices.append(i)
                logger.warning(f"[train] Skip unknown-error sample idx={i}, err={repr(e)}")
                continue

            save_path_train = save_dir_train / f"sample_{i}.pkl"
            with open(save_path_train, "wb") as f:
                pickle.dump(sample_train, f)
            valid_train_indices.append(i)

        # Validation 데이터 저장(깨진 항목 skip)
        for i in tqdm(range(len(val_data)), desc="Saving valid samples"):
            try:
                sample_val = val_data[i]   # (feature_dict, target_dict)
            except load_errors as e:
                skipped_val_indices.append(i)
                token = None
                try:
                    token = getattr(val_data, "tokens", None)
                    token = token[i] if token is not None and i < len(token) else None
                except Exception:
                    pass
                logger.warning(f"[val] Skip corrupted sample idx={i}, token={token}, err={repr(e)}")
                continue
            except Exception as e:
                skipped_val_indices.append(i)
                logger.warning(f"[val] Skip unknown-error sample idx={i}, err={repr(e)}")
                continue

            save_path_val = save_dir_val / f"sample_{i}.pkl"
            with open(save_path_val, "wb") as f:
                pickle.dump(sample_val, f)
            valid_val_indices.append(i)

        logger.info(f"Train: kept {len(valid_train_indices)} / {len(train_data)} (skipped {len(skipped_train_indices)})")
        logger.info(f"Val:   kept {len(valid_val_indices)} / {len(val_data)} (skipped {len(skipped_val_indices)})")

        # >>> 이후 학습에서도 같은 깨진 샘플을 피하고 싶다면, Subset으로 교체
        if skip_corrupted:
            if len(valid_train_indices) > 0:
                train_data = Subset(train_data, valid_train_indices)
            else:
                logger.error("No valid train samples remain after skipping. Check your cache/data.")
            if len(valid_val_indices) > 0:
                val_data = Subset(val_data, valid_val_indices)
            else:
                logger.error("No valid val samples remain after skipping. Check your cache/data.")

        breakpoint()  # 원한다면 유지
    
    
    
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))
    # breakpoint()
    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
