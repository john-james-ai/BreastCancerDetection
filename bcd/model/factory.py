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
# Created    : Monday February 5th 2024 07:09:37 pm                                                #
# Modified   : Wednesday February 7th 2024 09:05:17 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Transfer Learning Model Centre"""
from abc import ABC, abstractmethod
from typing import Callable

import keras
import tensorflow as tf
from keras import layers


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=not-callable
# ------------------------------------------------------------------------------------------------ #
#                                ABSTRACT MODEL FACTORY                                            #
# ------------------------------------------------------------------------------------------------ #
class AbstractModelFactory(ABC):
    """Abstract Model Factory

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

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the model architecture version."""

    @abstractmethod
    def create_model(
        self, alias: str, pretrained_model: tf.keras.Model, preprocess_input: Callable
    ) -> tf.keras.Model:
        """Builds a CNN model on the designated pre-trained model."""

    @property
    def augmentation(self) -> tf.Tensor:
        """Performs random horizontal flip and rotation of images.

        Args:
            inputs (tf.Tensor): Input data
        """
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
            ],
            name="data_augmentation",
        )

    def create_densenet(self) -> tf.keras.Model:
        alias = "DenseNet201"
        pretrained_model = tf.keras.applications.densenet.DenseNet201(include_top=False)
        pretrained_model.trainable = False
        preprocess_input = tf.keras.applications.densenet.preprocess_input
        return self.create_model(
            alias=alias,
            pretrained_model=pretrained_model,
            preprocess_input=preprocess_input,
        )

    def create_resnet(self) -> tf.keras.Model:
        alias = "ResNet152V2"
        pretrained_model = tf.keras.applications.resnet_v2.ResNet152V2(
            include_top=False
        )
        pretrained_model.trainable = False
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        return self.create_model(
            alias=alias,
            pretrained_model=pretrained_model,
            preprocess_input=preprocess_input,
        )

    def create_inception(self) -> tf.keras.Model:
        alias = "InceptionV3"
        pretrained_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False
        )
        pretrained_model.trainable = False
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        return self.create_model(
            alias=alias,
            pretrained_model=pretrained_model,
            preprocess_input=preprocess_input,
        )

    def create_efficientnet(self) -> tf.keras.Model:
        alias = "EfficientNetV2"
        pretrained_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False
        )
        pretrained_model.trainable = False
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        return self.create_model(
            alias=alias,
            pretrained_model=pretrained_model,
            preprocess_input=preprocess_input,
        )

    def create_inception_resnet(self) -> tf.keras.Model:
        alias = "InceptionResNetV2"
        pretrained_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False
        )
        pretrained_model.trainable = False
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
        return self.create_model(
            alias=alias,
            pretrained_model=pretrained_model,
            preprocess_input=preprocess_input,
        )

    def create_mobilenet(self) -> tf.keras.Model:
        alias = "MobileNetV2"
        pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False
        )
        pretrained_model.trainable = False
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        return self.create_model(
            alias=alias,
            pretrained_model=pretrained_model,
            preprocess_input=preprocess_input,
        )

    def create_xception(self) -> tf.keras.Model:
        alias = "Xception"
        pretrained_model = tf.keras.applications.xception.Xception(include_top=False)
        pretrained_model.trainable = False
        preprocess_input = tf.keras.applications.xception.preprocess_input
        return self.create_model(
            alias=alias,
            pretrained_model=pretrained_model,
            preprocess_input=preprocess_input,
        )


# ------------------------------------------------------------------------------------------------ #
#                                   MODEL FACTORY v1                                               #
# ------------------------------------------------------------------------------------------------ #
class ModelFactoryV1(AbstractModelFactory):
    """Factory for CNN Transfer Learning models

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
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            activation=activation,
        )
        self._pretrained_model = None

    @property
    def version(self) -> str:
        """Returns the model architecture version"""
        return "V1"

    def create_model(
        self, alias: str, pretrained_model: tf.keras.Model, preprocess_input: Callable
    ) -> tf.keras.Model:
        # Create the input
        inputs = tf.keras.Input(
            shape=self._input_shape, batch_size=None, name="input_layer"
        )
        # Perform base model specific preprocessing
        x = preprocess_input(x=inputs)
        # Augment the image data
        x = self.augmentation(x)
        # Feed base model
        x = pretrained_model(x)
        # Pooling for dimensionality reduction
        x = tf.keras.layers.GlobalAveragePooling2D(
            name=f"{alias.lower()}_global_average_pooling"
        )(x)
        # Add fully connected layers with dropout for regularization
        x = tf.keras.layers.Dense(
            1024, activation="relu", name=f"{alias.lower()}_dense_1"
        )(x)
        x = tf.keras.layers.Dropout(0.5, name=f"{alias.lower()}_dropout_1")(x)
        x = tf.keras.layers.Dense(
            1024, activation="relu", name=f"{alias.lower()}_dense_2"
        )(x)
        x = tf.keras.layers.Dropout(0.3, name=f"{alias.lower()}_dropout_2")(x)
        x = tf.keras.layers.Dense(
            512, activation="relu", name=f"{alias.lower()}_dense_3"
        )(x)

        # Add Layers for classification
        x = tf.keras.layers.Dense(
            128, activation="relu", name=f"{alias.lower()}_dense_4"
        )(x)
        outputs = layers.Dense(
            units=self._output_shape,
            activation=self._activation,
            name=f"{alias.lower()}_output_layer",
        )(x)
        # Create the model and add metadata
        model = tf.keras.Model(inputs, outputs)

        model.alias = alias
        model.version = self.version

        return model
