#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/base.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday February 11th 2024 06:30:03 pm                                               #
# Modified   : Monday February 12th 2024 03:56:28 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import tensorflow as tf

from bcd import DataClass
from bcd.model.base import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        Network                                                   #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Network:
    """Network network class"""

    name: str  # Combination of the network class name and the base model name
    model: tf.keras.Model  # The model to be trained.
    base_model: (
        BaseModel  # BaseModel object containing the underlying pretrained model.
    )
    architecture: str  # The name for the architecture. This is derived from the factory class name.
    register_model: bool = (
        True  # Provided to avoid registering excessively large models.
    )
    config: NetworkConfig  # Hyperparameters for the network configuration.


# ------------------------------------------------------------------------------------------------ #
#                                     Network Config                                               #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class NetworkConfig(DataClass):
    """Base class for network configurations."""

    activation: str = "sigmoid"


# ------------------------------------------------------------------------------------------------ #
#                                     NetworkFactory                                               #
# ------------------------------------------------------------------------------------------------ #
class NetworkFactory(ABC):
    """Base network factory"""

    @property
    def augmentation(self) -> tf.Tensor:
        """Performs random horizontal flip and rotation of images.

        Args:
            inputs (tf.Tensor): Input data
        """
        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
            ],
            name="data_augmentation",
        )

    @abstractmethod
    def create(self, base_model: BaseModel) -> tf.keras.Model:
        """Creates a CNN transfer learning model for the given base model.

        Args:
            base_model (BaseModel): Base model object containing the pretrained model
                and the model specific preprocessor.
        """
