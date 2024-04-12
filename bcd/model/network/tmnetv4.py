#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/tmnetv4.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 10th 2024 09:56:45 am                                             #
# Modified   : Wednesday April 3rd 2024 08:18:14 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""TMNetV4 Module"""
from dataclasses import dataclass

import tensorflow as tf

from bcd.model.network.base import Network, NetworkConfig, NetworkFactory
from bcd.model.pretrained import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        TMNetV4 Config                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class TMNetV4Config(NetworkConfig):
    """TMNetV4 configuration"""

    dense1: int = 4096
    l21: float = 0.0001
    dropout1: float = 0.5
    dense2: int = 4096
    l22: float = 0.0001
    dropout2: float = 0.5
    dense3: int = 1024
    l23: float = 0.0001
    dropout3: float = 0.5
    dense4: int = 1024
    l24: float = 0.0001
    dropout4: float = 0.5


# ------------------------------------------------------------------------------------------------ #
#                                       TMNetV4 FActory                                            #
# ------------------------------------------------------------------------------------------------ #
class TMNetV4Factory(NetworkFactory):
    """Factory for CNN TMNetV4 Transfer Learning model"""

    __name = "TMNetV4"

    def __init__(
        self,
        config: TMNetV4Config,
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
        # Add fully connected layers
        x = tf.keras.layers.Dense(
            self._config.dense1,
            activation="elu",
            kernel_regularizer=tf.keras.regularizers.l2(self._config.l21),
            name=f"{name}_dense_1",
        )(x)
        x = tf.keras.layers.Dropout(self._config.dropout1, name=f"{name}_dropout_1")(x)
        x = tf.keras.layers.Dense(
            self._config.dense2,
            activation="elu",
            kernel_regularizer=tf.keras.regularizers.l2(self._config.l22),
            name=f"{name}_dense_2",
        )(x)
        x = tf.keras.layers.Dropout(self._config.dropout2, name=f"{name}_dropout_2")(x)
        x = tf.keras.layers.Dense(
            self._config.dense3,
            activation="elu",
            kernel_regularizer=tf.keras.regularizers.l2(self._config.l23),
            name=f"{name}_dense_3",
        )(x)
        x = tf.keras.layers.Dropout(self._config.dropout3, name=f"{name}_dropout_3")(x)
        x = tf.keras.layers.Dense(
            self._config.dense4,
            activation="elu",
            kernel_regularizer=tf.keras.regularizers.l2(self._config.l24),
            name=f"{name}_dense_4",
        )(x)
        x = tf.keras.layers.Dropout(self._config.dropout4, name=f"{name}_dropout_4")(x)

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
