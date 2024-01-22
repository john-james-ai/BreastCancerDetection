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
# Created    : Sunday January 21st 2024 04:26:02 pm                                                #
# Modified   : Sunday January 21st 2024 09:35:19 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import pathlib
from dataclasses import dataclass, field
from typing import List

from bcd import DataClass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ModelConfig(DataClass):
    """Model Configuration"""

    name: str
    metrics: List = field(default_factory=["accuracy"])
    base_model_layer: int = 5
    loss: str = "binary_crossentropy"
    activation: str = "sigmoid"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ReduceLROnPlateauConfig(DataClass):
    """Reduce learning rate on plateau callback config."""

    min_delta: float = 0.0001
    monitor: str = "val_loss"
    factor: float = 0.1
    patience: int = 3
    verbose: int = 1
    min_lr: float = 1e-10


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataConfig(DataClass):
    """Data config"""

    full: bool = False
    batch_size: int = 32
    input_shape: tuple = (224, 224, 3)
    output_shape: int = 1
    train_dir: str = None
    test_dir: str = pathlib.Path("data/image/1_final/test/test/").with_suffix("")

    def __post_init__(self) -> None:
        self.batch_size = 32 if self.full is False else 64
        self.train_dir = (
            pathlib.Path("data/image/1_final/training/training/").with_suffix("")
            if self.full
            else pathlib.Path("data/image/1_final/training_10/training/").with_suffix(
                ""
            )
        )


# ------------------------------------------------------------------------------------------------ #
@dataclass
class EarlyStopConfig(DataClass):
    """Early stop config"""

    min_delta: float = 0.0001
    monitor: str = "val_loss"
    patience: int = 9
    restore_best_weights: bool = True
    verbose: int = 1


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ModelCheckpointCallbackConfig(DataClass):
    location: str = "models/"
    monitor: str = "val_loss"
    mode: str = "auto"
    save_weights_only: bool = False
    save_best_only: bool = True
    save_freq: str = "epoch"
    verbose: int = 1


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ExperimentConfig(DataClass):
    id: int = None
    full_dataset: bool = False
    model_config: ModelConfig = None
    data_config: DataConfig = None
    early_stop_config: EarlyStopConfig = None
