#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/tmnetv5.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 10th 2024 09:56:45 am                                             #
# Modified   : Friday March 15th 2024 06:33:28 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""TMNetV5 Module"""
from dataclasses import dataclass

import tensorflow as tf

from bcd.model.network.base import Network, NetworkConfig, NetworkFactory
from bcd.model.pretrained import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        TMNetV5 Config                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class TMNetV5Config(NetworkConfig):
    """TMNetV5 configuration"""

    dense1: int = 4096
    dense2: int = 4096
    dense3: int = 2048


# ------------------------------------------------------------------------------------------------ #
#                                       TMNetV5 FActory                                            #
# ------------------------------------------------------------------------------------------------ #
class TMNetV5Factory(NetworkFactory):
    """Factory for CNN TMNetV5 Transfer Learning model"""

    __name = "TMNetV5"

    def __init__(
        self,
        config: TMNetV5Config,
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
        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization(name=f"{name}_batch_normalization")(x)
        # Add fully connected layers
        x = tf.keras.layers.Dense(
            self._config.dense1,
            activation="elu",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            name=f"{name}_dense_1",
        )(x)
        x = tf.keras.layers.Dense(
            self._config.dense2,
            activation="elu",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            name=f"{name}_dense_2",
        )(x)
        x = tf.keras.layers.Dense(
            self._config.dense3,
            activation="elu",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            name=f"{name}_dense_3",
        )(x)

        # Add Layers for classification
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