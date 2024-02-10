#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/classifier.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 9th 2024 04:17:36 am                                                #
# Modified   : Friday February 9th 2024 10:05:23 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Classifier Network Module"""
from abc import ABC, abstractmethod
from typing import Union

import tensorflow as tf


# ------------------------------------------------------------------------------------------------ #
class ClassifierNetwork(ABC):
    """Abstract base lass for the Transfer Learning Classifier Network

    Args:
        name (str): The base model name
        output_shape (Union[tuple,int]]) Shape of output. Default = 1
        activation (str): Activation function providing class probabilities. Default = 'sigmoid'
    """

    def __init__(
        self,
        name: str,
        output_shape: Union[tuple, int] = 1,
        activation: str = "sigmoid",
    ) -> None:
        self._name = name
        self._output_shape = output_shape
        self._activation = activation

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_shape(self) -> int:
        return self._output_shape

    @property
    def activation(self) -> str:
        return self._activation

    @abstractmethod
    def apply_training_layers(self, features: tf.Tensor) -> tf.Tensor:
        """Applies the training layer to the input"""

    @abstractmethod
    def apply_classifier_layer(self, x: tf.Tensor) -> tf.Tensor:
        """Apply the classification layer"""


# ------------------------------------------------------------------------------------------------ #
class BCDClassifierV1(ClassifierNetwork):
    """Classifier Network V1.

    This network contains the following layers:
        - Global Average Pooling for dimensionality reduction
        - Two blocks containing:
            - fully connected layer with ReLU activations
            - dropout layer for regularization
        - Dense layer
        - Classifier layer with sigmoid activation,

    Args:
        name (str): The base model name
        output_shape (Union[tuple,int]]) Shape of output. Default = 1
        activation (str): Activation function providing class probabilities. Default = 'sigmoid'
    """

    def __init__(
        self,
        name: str,
        output_shape: Union[tuple, int] = 1,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__(name=name, output_shape=output_shape, activation=activation)

    def apply_training_layers(self, features: tf.Tensor) -> tf.Tensor:
        """Applies the training layer to the input"""

        # Pooling for dimensionality reduction
        x = tf.keras.layers.GlobalAveragePooling2D(
            name=f"{self.name.lower()}_global_average_pooling"
        )(features)

        # Add fully connected layers with dropout for regularization
        x = tf.keras.layers.Dense(
            1024, activation="relu", name=f"{self.name.lower()}_dense_1"
        )(x)
        x = tf.keras.layers.Dropout(0.5, name=f"{self.name.lower()}_dropout_1")(x)
        x = tf.keras.layers.Dense(
            1024, activation="relu", name=f"{self.name.lower()}_dense_2"
        )(x)
        x = tf.keras.layers.Dropout(0.3, name=f"{self.name.lower()}_dropout_2")(x)
        x = tf.keras.layers.Dense(
            512, activation="relu", name=f"{self.name.lower()}_dense_3"
        )(x)
        x = tf.keras.layers.Dense(
            128, activation="relu", name=f"{self.name.lower()}_dense_4"
        )(x)
        return x

    def apply_classifier_layer(self, x: tf.Tensor) -> tf.Tensor:
        """Apply the classification layer"""

        outputs = tf.keras.layers.Dense(
            units=self.output_shape,
            activation=self.activation,
            name=f"{self.name.lower()}_output_layer",
        )(x)

        return outputs
