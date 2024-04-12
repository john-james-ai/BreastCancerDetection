#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/zznetv3.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 14th 2024 05:13:52 am                                                #
# Modified   : Friday March 15th 2024 06:33:29 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""ZZNetV3 Module"""

from dataclasses import dataclass

import tensorflow as tf

from bcd.model.network.base import Network, NetworkConfig, NetworkFactory
from bcd.model.pretrained import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        ZZNetV3 Config                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ZZNetV3Config(NetworkConfig):
    """ZZNetV3 configuration"""

    description: str = "ZZNETV3: One dense layer followed by dropout"
    dense1: int = 1024
    dropout1: float = 0.6


# ------------------------------------------------------------------------------------------------ #
#                                       ZZNetV3 FActory                                            #
# ------------------------------------------------------------------------------------------------ #
class ZZNetV3Factory(NetworkFactory):
    """Factory for CNN ZZNetV3 Transfer Learning model

    Reference:
    [1] R. R and L. Kalaivani, “Breast Cancer Detection and Classification using Deeper
    Convolutional Neural Networks based on Wavelet Packet Decomposition Techniques,”
    In Review, preprint, Apr. 2021. doi: 10.21203/rs.3.rs-405990/v1.


    """

    __name = "ZZNetV3"

    def __init__(
        self,
        config: ZZNetV3Config,
    ) -> None:
        self._config = config

    def create(self, base_model: BaseModel) -> Network:
        """Creates a CNN transfer learning model for the given base model.

        Args:
            base_model (BaseModel): Base model object containing the pretrained model
                and the model specific preprocessor.
        """
        # Designate a model name that will be used to name runs.
        name = f"{self.__name}_{base_model.name}"
        # Create the input
        inputs = tf.keras.Input(
            shape=self._config.input_shape, batch_size=None, name=f"{name}_input_layer"
        )
        # Perform base model specific preprocessing
        x = base_model.preprocessor(x=inputs)
        # Feed base model
        x = base_model.model(x, training=False)

        # Pooling for dimensionality reduction
        x = tf.keras.layers.GlobalAveragePooling2D(
            name=f"{name}_global_average_pooling"
        )(x)

        # Dense, batch norm, followed by activation as per original paper
        x = tf.keras.layers.Dense(
            self._config.dense1, activation="relu", name=f"{name}_dense_1"
        )(x)

        # Dropout layer
        x = tf.keras.layers.Dropout(self._config.dropout1)(x)

        # Add Layer for classification
        outputs = tf.keras.layers.Dense(
            units=self._config.output_shape,
            activation=self._config.activation,
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
            config=self._config,
        )

        return network