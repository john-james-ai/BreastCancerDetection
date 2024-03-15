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
# Modified   : Thursday March 14th 2024 03:29:45 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import tensorflow as tf

from bcd.model.config import NetworkConfig
from bcd.model.pretrained import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        Network                                                   #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Network:
    """Network network class"""

    name: str = None  # Combination of the network class name and the base model name
    model: tf.keras.Model = None  # The model to be trained.
    base_model: (
        BaseModel  # BaseModel object containing the underlying pretrained model.
    ) = None
    architecture: str = (
        None  # The name for the architecture. This is derived from the factory class name.
    )
    register_model: bool = (
        True  # Provided to avoid registering excessively large models.
    )
    config: NetworkConfig = None  # Hyperparameters for the network configuration.

    def summary(self) -> None:
        self.model.summary()


# ------------------------------------------------------------------------------------------------ #
#                                     NetworkFactory                                               #
# ------------------------------------------------------------------------------------------------ #
class NetworkFactory(ABC):
    """Base network factory"""

    @abstractmethod
    def create(self, base_model: BaseModel) -> Network:
        """Creates a CNN transfer learning model for the given base model.

        Args:
            base_model (BaseModel): Base model object containing the pretrained model
                and the model specific preprocessor.
        """
