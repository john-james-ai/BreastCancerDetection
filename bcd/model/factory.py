#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/factory.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 10th 2024 09:56:45 am                                             #
# Modified   : Saturday February 10th 2024 10:36:38 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Model Factory Module"""
from abc import ABC, abstractmethod

import tensorflow as tf

from bcd.model.base import BaseModel


# ------------------------------------------------------------------------------------------------ #
#                                     ModelFactory                                                 #
# ------------------------------------------------------------------------------------------------ #
class ModelFactory(ABC):
    """Base Model Factory"""

    @property
    def augmentation(self) -> tf.Tensor:
        """Performs random horizontal flip and rotation of images.

        Args:
            inputs (tf.Tensor): Input data
        """
        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
            ],
            name="data_augmentation",
        )

    @abstractmethod
    def create(self, base_model: BaseModel) -> tf.keras.Model:
        """Creates a CNN transfer learning model for the given base model.

        Args:
            base_model (BaseModel): Base model object containing the pretrained model
                and the model specific preprocessor.
        """


# ------------------------------------------------------------------------------------------------ #
#                                     ShaneNetFactory                                                 #
# ------------------------------------------------------------------------------------------------ #
class ShainNetFactory(ModelFactory):
    """Factory for CNN ShainNet Transfer Learning model

    Models are comprised of a frozen pre-trained model upon which, the following layers are added:
    - Global Average Pooling Layer
    - Dense layer with 1024 nodes and ReLU activation
    - Dropout layer with rate = 0.5
    - Dense layer with 1024 nodes and ReLU activation
    - Dropout layer with rate - 0.3
    - Dense llayer with 512 nodes and ReLU activation
    - Dense layer with 128 nodes and ReLU activation
    - Dense layer with sigmoid activation

    Args:
        input_shape (tuple): Shape of input. Default = (224,224,3).
        output_shape (int): Shape of output. Default = 1.
        activation (str): Output activation. Default = 'sigmoid'.

    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        output_shape: int = 1,
        activation: str = "sigmoid",
    ) -> None:
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._activation = activation

    def create(self, base_model: BaseModel) -> tf.keras.Model:
        """Creates a CNN transfer learning model for the given base model.

        Args:
            base_model (BaseModel): Base model object containing the pretrained model
                and the model specific preprocessor.
        """
        # Designate a model name that will prepend layer names.
        name = f"ShainNet_{base_model.name}"
        # Create the input
        inputs = tf.keras.Input(
            shape=self._input_shape, batch_size=None, name=f"{name}_input_layer"
        )
        # Perform base model specific preprocessing
        x = base_model.preprocessor(x=inputs)
        # Augment the image data
        x = self.augmentation(x)
        # Feed base model
        x = base_model.model(x)
        # Pooling for dimensionality reduction
        x = tf.keras.layers.GlobalAveragePooling2D(
            name=f"{name}_global_average_pooling"
        )(x)
        # Add fully connected layers with dropout for regularization
        x = tf.keras.layers.Dense(1024, activation="relu", name=f"{name}_dense_1")(x)
        x = tf.keras.layers.Dropout(0.5, name=f"{name}_dropout_1")(x)
        x = tf.keras.layers.Dense(1024, activation="relu", name=f"{name}_dense_2")(x)
        x = tf.keras.layers.Dropout(0.3, name=f"{name}_dropout_2")(x)
        x = tf.keras.layers.Dense(512, activation="relu", name=f"{name}_dense_3")(x)

        # Add Layers for classification
        x = tf.keras.layers.Dense(128, activation="relu", name=f"{name}_dense_4")(x)
        outputs = tf.keras.layers.Dense(
            units=self._output_shape,
            activation=self._activation,
            name=f"{name}_output_layer",
        )(x)
        # Create the model and add metadata
        model = tf.keras.Model(inputs, outputs)
        model.alias = name
        model.base_model = base_model.name
        model.architecture = "ShainNet"
        return model


# ------------------------------------------------------------------------------------------------ #
