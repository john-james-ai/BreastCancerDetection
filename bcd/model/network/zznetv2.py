#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/zznet2.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 14th 2024 05:12:02 am                                                #
# Modified   : Thursday March 14th 2024 03:29:46 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""ZZNetV2 Module"""

from dataclasses import dataclass

import tensorflow as tf

from bcd.model.network.base import Network, NetworkConfig, NetworkFactory
from bcd.model.pretrained import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        ZZNetV2 Config                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ZZNetV2Config(NetworkConfig):
    """ZZNetV2 configuration"""

    description: str = "ZZNETV2: Two dense layers followed by dropout"
    dense1: int = 1024
    dense2: int = 512
    dropout2: float = 0.5


# ------------------------------------------------------------------------------------------------ #
#                                       ZZNetV2 FActory                                            #
# ------------------------------------------------------------------------------------------------ #
class ZZNetV2Factory(NetworkFactory):
    """Factory for CNN ZZNetV2 Transfer Learning model

    Reference:
    [1] R. R and L. Kalaivani, “Breast Cancer Detection and Classification using Deeper
    Convolutional Neural Networks based on Wavelet Packet Decomposition Techniques,”
    In Review, preprint, Apr. 2021. doi: 10.21203/rs.3.rs-405990/v1.


    """

    __name = "ZZNetV2"

    def __init__(
        self,
        config: ZZNetV2Config,
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
        x = base_model.model(x)

        # Pooling for dimensionality reduction
        x = tf.keras.layers.GlobalAveragePooling2D(
            name=f"{name}_global_average_pooling"
        )(x)

        # Dense, batch norm, followed by activation as per original paper
        x = tf.keras.layers.Dense(
            self._config.dense1, activation="relu", name=f"{name}_dense_1"
        )(x)

        # Dense, batch norm, followed by activation as per original paper
        x = tf.keras.layers.Dense(
            self._config.dense2, activation="relu", name=f"{name}_dense_2"
        )(x)

        # Dropout layer
        x = tf.keras.layers.Dropout(self._config.dropout2)(x)

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
