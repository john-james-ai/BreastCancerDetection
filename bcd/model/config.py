#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/config.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday February 12th 2024 12:36:45 pm                                               #
# Modified   : Sunday February 18th 2024 04:18:03 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Model and Experiment Configuration Module"""
from dataclasses import dataclass

from bcd import DataClass
from bcd.utils.hash import dict_hash


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Config(DataClass):
    """Base class for configuration objects"""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ProjectConfig(Config):
    mode: str
    name: str = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"Breast-Cancer-Detection-{self.mode}"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TrainConfig(Config):
    epochs: int
    learning_rate: float = 1e-4
    loss: str = "binary_crossentropy"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class NetworkConfig(Config):
    activation: str = "sigmoid"
    input_shape: tuple = (224, 224, 3)
    output_shape: int = 1


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetConfig(Config):
    mode: str
    name: str = None
    batch_size: int = 32
    labels: str = "inferred"
    color_mode: str = "rgb"
    image_size: tuple = (224, 224)
    shuffle: bool = True
    validation_split: float = 0.2
    interpolation: str = "bilinear"
    seed: int = 555

    def __post_init__(self) -> None:
        if "dev" in self.mode.lower():
            self.name = "CBIS-DDSM-10"
        elif "stage" in self.mode.lower():
            self.name = "CBIS-DDSM-30"
        else:
            self.name = "CBIS-DDSM"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class CheckPointConfig(Config):
    directory: str = None
    monitor: str = "val_accuracy"
    verbose: int = 1
    save_best_only: bool = True
    save_weights_only: bool = False
    mode: str = "auto"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class EarlyStopConfig(Config):
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    patience: int = 10
    restore_best_weights: bool = True
    verbose: int = 1


# ------------------------------------------------------------------------------------------------ #
@dataclass
class LearningRateScheduleConfig(Config):
    min_lr: float = 1e-8
    max_lr: float = 1e-1
    epochs: int = 10
    steps_size: int = 4


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ExperimentConfig(Config):
    project: ProjectConfig
    dataset: DatasetConfig
    train: TrainConfig
    network: NetworkConfig
    checkpoint: CheckPointConfig
    early_stop: EarlyStopConfig
    learning_rate_schedule: LearningRateScheduleConfig

    def as_dict(self) -> dict:
        config = super().as_dict()
        config["hash"] = dict_hash(config)
        return config
