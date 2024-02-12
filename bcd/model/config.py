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
# Modified   : Monday February 12th 2024 01:44:28 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Model and Experiment Configuration Module"""
import pathlib
from dataclasses import dataclass

from bcd import DataClass
from bcd.utils.hash import dict_hash


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ProjectConfig(DataClass):
    mode: str
    name: str = None

    def __post_init__(self) -> None:
        self.name = f"Breast-Cancer-Detection-{self.mode}"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TrainConfig(DataClass):
    epochs: int
    learning_rate: float = 1e-4
    loss: str = "binary_crossentropy"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class NetworkConfig(DataClass):
    activation: str = "sigmoid"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetConfig(DataClass):
    mode: str
    dataset: str = None
    batch_size: int = 32
    input_shape: tuple = (224, 224, 3)
    output_shape: int = 1
    train_dir: str = None

    def __post_init__(self) -> None:
        datasets = {
            "Development": {
                "name": "CBIS-DDSM_10",
                "directory": "data/image/1_final/training_10/training/",
            },
            "Stage": {
                "name": "CBIS-DDSM_30",
                "directory": "data/image/1_final/training_30/training/",
            },
            "Production": {
                "name": "CBIS-DDSM",
                "directory": "data/image/1_final/training/training/",
            },
        }
        self.dataset = datasets[self.mode]["name"]
        self.train_dir = datasets[self.mode]["directory"]
        self.batch_size = 64 if self.mode == "Production" else 32


# ------------------------------------------------------------------------------------------------ #
@dataclass
class CheckPointConfig(DataClass):
    monitor: str = "val_accuracy"
    verbose: int = 1
    save_best_only: bool = True
    save_weights_only: bool = False
    mode: str = "auto"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class EarlyStopConfig(DataClass):
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    patience: int = 10
    restore_best_weights: bool = True
    verbose: int = 1


# ------------------------------------------------------------------------------------------------ #
@dataclass
class LearningRateScheduleConfig(DataClass):
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    factor: float = 0.5
    patience: int = 3
    restore_best_weights: bool = True
    verbose: int = 1
    mode: str = "auto"
    min_lr = 1e-8


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Config(DataClass):
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
