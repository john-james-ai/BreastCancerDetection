#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/nlnetv6.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 10th 2024 09:56:45 am                                             #
# Modified   : Friday March 15th 2024 06:33:29 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""NLNetV6 Module"""
from dataclasses import dataclass

import tensorflow as tf

from bcd.model.network.base import Network, NetworkConfig, NetworkFactory
from bcd.model.pretrained import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        NLNetV6 Config                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class NLNetV6Config(NetworkConfig):
    """NLNetV6 configuration"""

    description: str = "Batchnorm x 3 After Activation"
    dense1: int = 4096
    dense2: int = 4096
    dense3: int = 1024


# ------------------------------------------------------------------------------------------------ #
#                                       NLNetV6 FActory                                            #
# ------------------------------------------------------------------------------------------------ #
class NLNetV6Factory(NetworkFactory):
    """Factory for CNN NLNetV6 Transfer Learning model"""

    __name = "NLNetV6"

    def __init__(
        self,
        config: NLNetV6Config,
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

        # Dense, activation, then batch norm
        x = tf.keras.layers.Dense(self._config.dense1, name=f"{name}_dense_1")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_batch_norm_1")(x)

        # Dense, activation, then batch norm
        x = tf.keras.layers.Dense(self._config.dense2, name=f"{name}_dense_2")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_batch_norm_2")(x)

        # Dense, activation, then batch norm
        x = tf.keras.layers.Dense(self._config.dense3, name=f"{name}_dense_3")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_batch_norm_3")(x)

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
