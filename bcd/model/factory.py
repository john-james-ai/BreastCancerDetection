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
# Modified   : Wednesday February 7th 2024 12:08:48 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Transfer Learning Model Centre"""
from abc import ABC, abstractmethod

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

        self._pretrained_model = None
        self._preprocess_input = lambda x: x

    def reset(self) -> None:
        self._pretrained_model = None
        self._preprocess_input = lambda x: x

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the model architecture version."""

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

    @property
    def pretrained_model(self) -> tf.keras.Model:
        return self._pretrained_model

    @abstractmethod
    def create_model(self) -> tf.keras.Model:
        """Builds a CNN model on the designated pre-trained model."""

    def create_densenet(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.densenet.DenseNet201(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.densenet.preprocess_input
        model = self.create_model()
        model.alias = "DenseNet201"
        return model

    def create_resnet(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.resnet_v2.ResNet152V2(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        model = self.create_model()
        model.alias = "ResNet152V2"
        return model

    def create_inception(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        model = self.create_model()
        model.alias = "InceptionV3"
        return model

    def create_efficientnet(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        model = self.create_model()
        model.alias = "EfficientNetV2"
        return model

    def create_inception_resnet(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = (
            tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
                include_top=False
            )
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = (
            tf.keras.applications.inception_resnet_v2.preprocess_input
        )
        model = self.create_model()
        model.alias = "InceptionResNetV2"
        return model

    def create_mobilenet(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        model = self.create_model()
        model.alias = "MobileNetV2"
        return model

    def create_xception(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.xception.Xception(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.xception.preprocess_input
        model = self.create_model()
        model.alias = "Xception"
        return model

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs preprocessing on inputs. Model-specific processing overridden in subclasses.

        Args:
            inputs (tf.Tensor): Model inputs
        """
        return self._preprocess_input(inputs)


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

    def create_model(self) -> tf.keras.Model:
        # Confirm base model have been created.
        assert self._pretrained_model is not None, "Base model must be created."
        # Create the input
        inputs = tf.keras.Input(
            shape=self._input_shape, batch_size=None, name="input_layer"
        )
        # Perform base model specific preprocessing
        x = self.preprocess_input(inputs=inputs)
        # Augment the image data
        x = self.augmentation(x)
        # Feed base model
        x = self.pretrained_model(x)
        # Pooling for dimensionality reduction
        x = tf.keras.layers.GlobalAveragePooling2D(name="bcd_global_average_pooling")(x)
        # Add fully connected layers with dropout for regularization
        x = tf.keras.layers.Dense(1024, activation="relu", name="bcd_dense_1")(x)
        x = tf.keras.layers.Dropout(0.5, name="bcd_dropout_1")(x)
        x = tf.keras.layers.Dense(1024, activation="relu", name="bcd_dense_2")(x)
        x = tf.keras.layers.Dropout(0.3, name="bcd_dropout_2")(x)
        x = tf.keras.layers.Dense(512, activation="relu", name="bcd_dense_3")(x)

        # Add Layers for classification
        x = tf.keras.layers.Dense(128, activation="relu", name="bcd_dense_4")(x)
        outputs = layers.Dense(
            units=self._output_shape,
            activation=self._activation,
            name="bcd_output_layer",
        )(x)
        # Create the model
        model = tf.keras.Model(inputs, outputs)

        model.version = self.version

        return model
