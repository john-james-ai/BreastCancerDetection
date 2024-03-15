#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/ensemble.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday March 14th 2024 03:19:11 pm                                                #
# Modified   : Thursday March 14th 2024 04:15:33 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Ensemble Module"""
from dataclasses import dataclass

import tensorflow as tf

from bcd.model.config import NetworkConfig
from bcd.model.network.base import Network


# ------------------------------------------------------------------------------------------------ #
#                                        ENSEMBLE CONFIG                                           #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class EnsembleConfig(NetworkConfig):
    """Ensemble configuration"""

    description: str = None
    dense1: int = 16
    dropout1: float = 0.3


# ------------------------------------------------------------------------------------------------ #
#                                       ENSEMBLE FACTORY                                           #
# ------------------------------------------------------------------------------------------------ #
class EnsembleFactory:
    """Creates an ensemble network"""

    __name = "Ensemble"

    def __init__(
        self,
        config: EnsembleConfig,
    ) -> None:
        self._config = config

    def create(self, networks: list) -> Network:

        self._config.description = ",".join([network.name for network in networks])

        # Create the input
        inputs = tf.keras.Input(
            shape=self._config.input_shape,
            batch_size=None,
            name=f"{self.__name}_input_layer",
        )

        # Obtain output for each input network model
        outputs = [network.model(inputs) for network in networks]

        # Average outputs
        x = tf.keras.layers.Average()(outputs)

        # Add fully connected layers
        x = tf.keras.layers.Dense(
            self._config.dense1, activation="relu", name=f"{self.__name}_dense_1"
        )(x)

        # Add Drop out layer
        x = tf.keras.layers.Dropout(
            self._config.dropout1, name=f"{self.__name}_dropout_1"
        )(x)

        # Add Layers for classification
        output = tf.keras.layers.Dense(
            units=self._config.output_shape,
            activation=self._config.activation,
            name=f"{self.__name}_output_layer",
        )(x)

        # Create the model
        model = tf.keras.Model(inputs, output)

        # Create the network
        network = Network(
            name=self.__name,
            model=model,
            base_model=None,
            architecture="ensemble",
            register_model=False,
            config=self._config,
        )

        return network
