#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/convnet.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 10th 2024 09:56:45 am                                             #
# Modified   : Thursday March 14th 2024 03:29:46 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""ConvNet Module"""
from dataclasses import dataclass

import tensorflow as tf

from bcd.model.network.base import Network, NetworkConfig, NetworkFactory
from bcd.model.pretrained import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                        ConvNet Config                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ConvNetConfig(NetworkConfig):
    """ConvNet configuration"""

    conv1_filter: int = 64
    conv1_kernel: int = 3
    maxpool1: tuple = (2, 2)
    conv2_filter: int = 128
    conv2_kernel: int = 3
    maxpool2: tuple = (2, 2)
    conv3_filter: int = 256
    conv3_kernel: int = 3
    maxpool3: tuple = (2, 2)
    dense: int = 1024
    l2: float = 0.01
    activation: str = "linear"


# ------------------------------------------------------------------------------------------------ #
#                                       ConvNet FActory                                            #
# ------------------------------------------------------------------------------------------------ #
class ConvNetFactory(NetworkFactory):
    """Factory for CNN ConvNet Transfer Learning model

    Reference:
    [1] T. Mahmood, J. Li, Y. Pei, and F. Akhtar, “An Automated In-Depth Feature Learning Algorithm
    for Breast Abnormality Prognosis and Robust Characterization from Mammography Images Using
    Deep Transfer Learning,” Biology, vol. 10, no. 9, p. 859, Sep. 2021, doi: 10.3390/biology10090859.

    """

    __name = "ConvNet"

    def __init__(
        self,
        config: ConvNetConfig,
    ) -> None:
        self._config = config

    def create(self, base_model: BaseModel) -> Network:
        """Creates the ConvNet SVM Model

        Args:
            base_model (BaseModel): Not used for this model.
        """
        # Designate a model name that will be used to name runs.
        name = f"{self.__name}_{base_model.name}"
        # Create the input
        inputs = tf.keras.Input(
            shape=self._config.input_shape, batch_size=None, name=f"{name}_input_layer"
        )
        # Resize and rescale the input
        x = tf.keras.layers.Resizing(
            height=self._config.input_shape[0],
            width=self._config.input_shape[1],
            name=f"{name}_resizing",
        )(inputs)
        x = tf.keras.layers.Rescaling(
            1.0 / 255,
            input_shape=(self._config.input_shape[0], self._config.input_shape[1], 3),
            name=f"{name}_rescaling",
        )(x)

        # Block 1: 2D Convolution with 64 filters and 3x3 kernel
        x = tf.keras.layers.Conv2D(
            filters=self._config.conv1_filter,
            kernel_size=self._config.conv1_kernel,
            name=f"{name}_conv1",
        )(x)
        # Max pooling with 2x2 kernel
        x = tf.keras.layers.MaxPooling2D(
            pool_size=self._config.maxpool1,
            name=f"{name}_maxpool1",
        )(x)

        # Block 2: 2D Convolution with 128 filters and 3x3 kernel
        x = tf.keras.layers.Conv2D(
            filters=self._config.conv2_filter,
            kernel_size=self._config.conv2_kernel,
            name=f"{name}_conv2",
        )(x)
        # Max pooling with 2x2 kernel
        x = tf.keras.layers.MaxPooling2D(
            pool_size=self._config.maxpool2,
            name=f"{name}_maxpool2",
        )(x)

        # Block 3: 2D Convolution with 256 filters and 3x3 kernel
        x = tf.keras.layers.Conv2D(
            filters=self._config.conv3_filter,
            kernel_size=self._config.conv3_kernel,
            name=f"{name}_conv3",
        )(x)
        # Max pooling with 2x2 kernel
        x = tf.keras.layers.MaxPooling2D(
            pool_size=self._config.maxpool3,
            name=f"{name}_maxpool3",
        )(x)

        # Flatten tensor
        x = tf.keras.layers.Flatten(name=f"{name}_flatten")(x)

        # Fully connected llayer
        x = tf.keras.layers.Dense(
            self._config.dense, activation="relu", name=f"{name}_dense"
        )(x)

        # Add SVM Classifier
        outputs = tf.keras.layers.Dense(
            units=self._config.output_shape,
            kernel_regularizer=tf.keras.regularizers.l2(self._config.l2),
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
            architecture=self.__name,
            config=self._config,
        )

        return network
