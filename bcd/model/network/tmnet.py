#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/tmnet.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 10th 2024 09:56:45 am                                             #
# Modified   : Sunday February 18th 2024 11:38:36 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""TMNet Module"""
from dataclasses import dataclass

import tensorflow as tf

from bcd.model.network.base import Network, NetworkConfig, NetworkFactory
from bcd.model.pretrained import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        TMNet Config                                              #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class TMNetConfig(NetworkConfig):
    """TMNet configuration"""

    dense1: int = 1024
    dense2: int = 1024


# ------------------------------------------------------------------------------------------------ #
#                                       TMNet FActory                                              #
# ------------------------------------------------------------------------------------------------ #
class TMNetFactory(NetworkFactory):
    """Factory for CNN TMNet Transfer Learning model [1]_

    Models are comprised of a frozen pre-trained model upon which, the following layers are added:
    - Global Average Pooling Layer
    - Batch Normalization Layer
    - Dense layer with 1024 (default) nodes and ReLU activation
    - Dense layer with 1024 (default) nodes and ReLU activation
    - Dense layer with sigmoid activation

    Args:
        input_shape (tuple): Shape of input. Default = (224,224,3).
        output_shape (int): Shape of output. Default = 1.
        activation (str): Output activation. Default = 'sigmoid'.

    Reference:
    .. [1] T. Mahmood, J. Li, Y. Pei, and F. Akhtar, “An Automated In-Depth Feature Learning Algorithm
    for Breast Abnormality Prognosis and Robust Characterization from Mammography Images
    Using Deep Transfer Learning,” Biology, vol. 10, no. 9, p. 859, Sep. 2021,
    doi: 10.3390/biology10090859.


    """

    __name = "TMNet"

    def __init__(
        self,
        config: TMNetConfig,
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
        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization(name=f"{name}_batch_normalization")(x)
        # Add fully connected layers
        x = tf.keras.layers.Dense(
            self._config.dense1, activation="relu", name=f"{name}_dense_1"
        )(x)
        x = tf.keras.layers.Dense(
            self._config.dense1, activation="relu", name=f"{name}_dense_2"
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
