from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Dict, Union, List
import torch
import pytorch_lightning as pl

from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

import numpy as np

class AbstractAgent(torch.nn.Module, ABC):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self,
        requires_scene: bool = False,
    ):
        super().__init__()
        self.requires_scene = requires_scene

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """
        pass

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """
        pass

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent.
        :param features: Dictionary of features.
        :return: Dictionary of predictions.
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of target builders.
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of feature builders.
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # breakpoint()
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).numpy()

        # extract trajectory
        return Trajectory(poses)

    def compute_trajectory_law_gw(self, agent_input: AgentInput, scene: Scene, cfg):
        # TODO: law는 이 코드를 사용
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
            
        for target_builder in self.get_target_builders():
            targets.update(target_builder.compute_targets(scene, cfg))
        # breakpoint()
        ###########################################################
        # TODO: 여기서 input을 위한 배치 dimension 추가하면 될 듯 - 250904
        
        # add batch dimension
        # features = {k: v.unsqueeze(0) for k, v in features.items()}






        # breakpoint()

        ###########################################################
        # forward pass
        with torch.no_grad():
            predictions = self.forward(features, targets)
            # breakpoint()
            poses = predictions.squeeze(0).detach().cpu().numpy()

        # TODO: 경로를 다시 역변환. 하고나서 heading이랑 경로 viz 한번 해보기
        # ---------- rotate BACK to the original frame ----------
        # We earlier did (x', y') = (-y, x), i.e., +90°
        # Inverse is (x, y) = ( y', -x' ), i.e., -90°
        R_inv = np.array([[0.0, 1.0],
                        [-1.0, 0.0]], dtype=np.float32)  # -90° rotation

        # 1) XY 좌표 역회전
        xy = poses[:, :2] @ R_inv.T  # (T,2)
        poses[:, :2] = xy

        # 2) heading(요)이 포함되어 있으면 π/2를 빼서 원복
        #    (학습 때 +π/2 했으므로 역변환은 -π/2)
        if poses.shape[1] >= 3:
            poses[:, 2] = poses[:, 2] - np.pi / 2.0

        # extract trajectory
        return Trajectory(poses)

    def compute_trajectory_bevplanner_gw(self, agent_input: AgentInput, scene: Scene, cfg):
        # TODO: law는 이 코드를 사용
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        # breakpoint()
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
            
        for target_builder in self.get_target_builders():
            targets.update(target_builder.compute_targets(scene, cfg))
        # breakpoint()
        ###########################################################
        # TODO: 여기서 input을 위한 배치 dimension 추가하면 될 듯 - 250904
        
        # add batch dimension
        # features = {k: v.unsqueeze(0) for k, v in features.items()}






        # breakpoint()

        ###########################################################
        # forward pass
        # breakpoint()
        with torch.no_grad():
            predictions = self.forward(features, targets)
            # breakpoint()
            poses = predictions[0]['pred_ego_traj']['prediction_w_heading'].squeeze(0).detach().cpu().numpy()
            # poses = predictions.squeeze(0).detach().cpu().numpy()
        # TODO: 경로를 다시 역변환. 하고나서 heading이랑 경로 viz 한번 해보기

        # ---------- rotate BACK to the original frame ----------
        # TODO: bev planner는 돌아가있음.
        # R_inv = np.array([[0.0, 1.0],
        #                 [-1.0, 0.0]], dtype=np.float32)  # -90° rotation

        # xy = poses[:, :2] @ R_inv.T  # (T,2)
        # poses[:, :2] = xy

        # if poses.shape[1] >= 3:
        #     poses[:, 2] = poses[:, 2] - np.pi / 2.0
        # -------------------------------------------------------
        return Trajectory(poses)

    def compute_trajectory_ltf_gw(self, agent_input: AgentInput, scene: Scene, cfg):
        # TODO: law는 이 코드를 사용
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
            
        for target_builder in self.get_target_builders():
            targets.update(target_builder.compute_targets(scene, cfg))
        # breakpoint()
        ###########################################################
        # TODO: 여기서 input을 위한 배치 dimension 추가하면 될 듯 - 250904
        
        # add batch dimension
        # features = {k: v.unsqueeze(0) for k, v in features.items()}






        # breakpoint()

        ###########################################################
        # forward pass
        with torch.no_grad():
            predictions = self.forward(features, targets)
            # breakpoint()
            poses = predictions["trajectory"].squeeze(0).detach().cpu().numpy()# [:, :6, :3]    # 8,3
            # poses = predictions.squeeze(0).detach().cpu().numpy()

        # TODO: 경로를 다시 역변환. 하고나서 heading이랑 경로 viz 한번 해보기
        # ---------- rotate BACK to the original frame ----------
        # We earlier did (x', y') = (-y, x), i.e., +90°
        # Inverse is (x, y) = ( y', -x' ), i.e., -90°
        R_inv = np.array([[0.0, 1.0],
                        [-1.0, 0.0]], dtype=np.float32)  # -90° rotation

        # 1) XY 좌표 역회전
        xy = poses[:, :2] @ R_inv.T  # (T,2)
        poses[:, :2] = xy

        # 2) heading(요)이 포함되어 있으면 π/2를 빼서 원복
        #    (학습 때 +π/2 했으므로 역변환은 -π/2)
        if poses.shape[1] >= 3:
            poses[:, 2] = poses[:, 2] - np.pi / 2.0

        # extract trajectory
        return Trajectory(poses)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]]]:
        """
        Returns the optimizers that are used by thy pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks that are used during training.
        See navsim.planning.training.callbacks for examples.
        """
        return []
