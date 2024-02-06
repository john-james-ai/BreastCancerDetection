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
# Modified   : Tuesday February 6th 2024 12:38:49 am                                               #
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
        num_classes (int): Number of output classes. Default = 2.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        output_shape: int = 1,
        activation: str = "sigmoid",
        num_classes: int = 2,
    ) -> None:
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._activation = activation
        self._num_classes = num_classes
        self._pretrained_model = None
        self._preprocess_input = lambda x: x

    def reset(self) -> None:
        self._pretrained_model = None
        self._preprocess_input = lambda x: x

    @property
    def augmentation(self) -> tf.Tensor:
        """Performs random horizontal flip and rotation of images.

        Args:
            inputs (tf.Tensor): Input data
        """
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.2),
            ],
            name="data_augmentation",
        )

    @property
    def pretrained_model(self) -> tf.keras.Model:
        return self._pretrained_model

    @abstractmethod
    def create_base_model(self) -> tf.keras.Model:
        """Creates a base image classification model"""

    @abstractmethod
    def create_densenet(self) -> tf.keras.Model:
        """Creates a DenseNet Model"""

    @abstractmethod
    def create_resnet(self) -> tf.keras.Model:
        """Creates a ResNet Model"""

    @abstractmethod
    def create_inception(self) -> tf.keras.Model:
        """Creates a Inception Model"""

    @abstractmethod
    def create_efficientnet(self) -> tf.keras.Model:
        """Creates a EfficientNet Model"""

    @abstractmethod
    def create_inception_resnet(self) -> tf.keras.Model:
        """Creates a Inception/ResNet Model"""

    @abstractmethod
    def create_mobilenet(self) -> tf.keras.Model:
        """Creates a MobileNet Model"""

    @abstractmethod
    def create_xception(self) -> tf.keras.Model:
        """Creates a Xception Model"""

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs preprocessing on inputs. Model-specific processing overridden in subclasses.

        Args:
            inputs (tf.Tensor): Model inputs
        """
        return self._preprocess_input(inputs)


# ------------------------------------------------------------------------------------------------ #
#                                   MODEL FACTORY v2                                               #
# ------------------------------------------------------------------------------------------------ #
class ModelFactoryV2(AbstractModelFactory):
    """Pre-trained models with two additional 4096 node dense layers and a sigmoid layer for classification.

    This implementation is based upon [1]_.

    Args:
        input_shape (tuple): Shape of input. Default = (224,224,3).
        units (int): Number of units in dense layer.
        output_shape (int): Shape of output. Default = 1.
        activation (str): Output activation. Default = 'sigmoid'.

    References:

    .. [1] A. Altameem, C. Mahanty, R. C. Poonia, A. K. J. Saudagar, and R. Kumar,
    “Breast Cancer Detection in Mammography Images Using Deep Convolutional Neural Networks and
    Fuzzy Ensemble Modeling Techniques,” Diagnostics, vol. 12, no. 8, p. 1812, Jul. 2022, doi:
    10.3390/diagnostics12081812.

    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        units: int = 4096,
        output_shape: int = 1,
        activation: str = "sigmoid",
        num_classes: int = 2,
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            activation=activation,
            num_classes=num_classes,
        )
        self._units = units
        self._pretrained_model = None

    def create_base_model(self) -> tf.keras.Model:
        self.reset()
        # Create the input layer
        inputs = tf.keras.Input(
            shape=self._input_shape, batch_size=None, name="input_layer"
        )
        # Augment the image data
        x = self.augmentation(inputs)
        # Perform image preprocessing
        x = tf.keras.layers.Rescaling(1.0 / 255, input_shape=self._input_shape)(x)
        # Create Convolutional Layer with 16 kernels
        x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        # Add max pooling layer
        x = tf.keras.layers.MaxPooling2D()(x)
        # Create Convolutional Layer with 32 kernels
        x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        # Add max pooling layer
        x = tf.keras.layers.MaxPooling2D()(x)
        # Create Convolutional Layer with 64 kernels
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        # Add max pooling layer and flatten output
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        # Create Dense Layer
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        # Produce prediction via sigmoid output layer
        outputs = tf.keras.layers.Dense(
            units=self._output_shape, activation=self._activation, name="output_layer"
        )(x)
        # Create the model
        model = tf.keras.Model(inputs, outputs)
        return model

    def create_densenet(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.densenet.DenseNet201(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.densenet.preprocess_input
        return self._create_model()

    def create_resnet(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.resnet_v2.ResNet152V2(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        return self._create_model()

    def create_inception(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        return self._create_model()

    def create_efficientnet(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        return self._create_model()

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
        return self._create_model()

    def create_mobilenet(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        return self._create_model()

    def create_xception(self) -> tf.keras.Model:
        self.reset()
        self._pretrained_model = tf.keras.applications.xception.Xception(
            include_top=False
        )
        self._pretrained_model.trainable = False
        self._preprocess_input = tf.keras.applications.xception.preprocess_input
        return self._create_model()

    def _create_model(self) -> tf.keras.Model:
        # Confirm base model have been created.
        assert self._pretrained_model is not None, "Base model must be created."
        # Create the input
        inputs = tf.keras.Input(
            shape=self._input_shape, batch_size=None, name="input_layer"
        )
        # Augment the image data
        x = self.augmentation(inputs)
        # Perform base model specific preprocessing
        x = self.preprocess_input(inputs=inputs)
        # Feed base model
        x = self.pretrained_model(x)
        # Add dense layers
        x = tf.keras.layers.Dense(
            self._units, activation="relu", name="dense_fine_tune_layer_1"
        )(x)
        x = tf.keras.layers.Dense(
            self._units, activation="relu", name="dense_fine_tune_layer_2"
        )(x)
        # Dimensionality reduction with global average pooling
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(
            x
        )
        # Produce prediction via sigmoid output layer
        outputs = tf.keras.layers.Dense(
            units=self._output_shape, activation=self._activation, name="output_layer"
        )(x)
        # Create the model
        model = tf.keras.Model(inputs, outputs)

        return model
