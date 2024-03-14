#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/zznet1.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 14th 2024 05:05:50 am                                                #
# Modified   : Thursday March 14th 2024 06:22:47 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""ZZNetV1 Module"""

from dataclasses import dataclass

import tensorflow as tf

from bcd.model.network.base import Network, NetworkConfig, NetworkFactory
from bcd.model.pretrained import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        ZZNetV1 Config                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ZZNetV1Config(NetworkConfig):
    """ZZNetV1 configuration"""

    description: str = "ZZNETV1: Three dense layers followed by dropout"
    dense1: int = 2048
    dense2: int = 1024
    dense3: int = 1024
    dropout3: float = 0.5


# ------------------------------------------------------------------------------------------------ #
#                                       ZZNetV1 FActory                                            #
# ------------------------------------------------------------------------------------------------ #
class ZZNetV1Factory(NetworkFactory):
    """Factory for CNN ZZNetV1 Transfer Learning model

    Reference:
    [1] R. R and L. Kalaivani, “Breast Cancer Detection and Classification using Deeper
    Convolutional Neural Networks based on Wavelet Packet Decomposition Techniques,”
    In Review, preprint, Apr. 2021. doi: 10.21203/rs.3.rs-405990/v1.


    """

    __name = "ZZNetV1"

    def __init__(
        self,
        config: ZZNetV1Config,
    ) -> None:
        self._config = config

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

        # Dense, batch norm, followed by activation as per original paper
        x = tf.keras.layers.Dense(
            self._config.dense3, activation="relu", name=f"{name}_dense_3"
        )(x)
        x = tf.keras.layers.Dropout(self._config.dropout3)(x)

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
