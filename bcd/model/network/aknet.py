#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/aknet.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 10th 2024 09:56:45 am                                             #
# Modified   : Monday February 12th 2024 12:47:12 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""AKNet Module"""
from dataclasses import dataclass

import tensorflow as tf

from bcd.model.base import BaseModel
from bcd.model.config import NetworkConfig
from bcd.model.network.base import Network, NetworkFactory


# ------------------------------------------------------------------------------------------------ #
#                                        TMNet Config                                              #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class AKNetConfig(NetworkConfig):
    """AKNet configuration"""

    dense1: int = 4096
    dense2: int = 4096


# ------------------------------------------------------------------------------------------------ #
#                                       AKNet FActory                                              #
# ------------------------------------------------------------------------------------------------ #
class AKNetFactory(NetworkFactory):
    """Factory for CNN AKNet Transfer Learning model

    Models are comprised of a frozen pre-trained model upon which, the following layers are added:
    - Global Average Pooling Layer
    - Dense layer with 4096 (default) nodes and ReLU activation
    - Dense layer with 4096 (default) nodes and ReLU activation
    - Dense layer with sigmoid activation

    Args:
        input_shape (tuple): Shape of input. Default = (224,224,3).
        output_shape (int): Shape of output. Default = 1.
        activation (str): Output activation. Default = 'sigmoid'.

    Source:


    """

    __name = "AKNet"

    def __init__(
        self,
        config: AKNetConfig,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        output_shape: int = 1,
    ) -> None:
        self._config = config
        self._input_shape = input_shape
        self._output_shape = output_shape

    def create(self, base_model: BaseModel) -> tf.keras.Model:
        """Creates a CNN transfer learning model for the given base model.

        Args:
            base_model (BaseModel): Base model object containing the pretrained model
                and the model specific preprocessor.
        """
        # Designate a model name that will be used to name runs.
        name = f"{self.__name}_{base_model.name}"
        # Create the input
        inputs = tf.keras.Input(
            shape=self._input_shape, batch_size=None, name=f"{name}_input_layer"
        )
        # Perform base model specific preprocessing
        x = base_model.preprocessor(x=inputs)
        # Augment the image data
        x = self.augmentation(x)
        # Feed base model
        x = base_model.model(x)
        # Flatten the output from the base model.
        x = tf.keras.layers.Flatten(name=f"{name}_flatten")(x)
        # Add fully connected layers
        x = tf.keras.layers.Dense(
            self._config.dense1, activation="relu", name=f"{name}_dense_1"
        )(x)
        x = tf.keras.layers.Dense(
            self._config.dense1, activation="relu", name=f"{name}_dense_2"
        )(x)

        # Add Layers for classification
        outputs = tf.keras.layers.Dense(
            units=self._output_shape,
            activation=self._config["activation"],
            name=f"{name}_output_layer",
        )(x)
        # Create the model
        model = tf.keras.Model(inputs, outputs)
        # Create the network
        name = self.__name + "_" + base_model.name
        network = Network(
            name=name,
            model=model,
            base_model=base_model.model,
            architecture=self.__name,
            register_model=False,
            config=self._config,
        )

        return network
