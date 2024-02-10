#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/factory_old.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday February 5th 2024 07:09:37 pm                                                #
# Modified   : Saturday February 10th 2024 10:41:16 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Transfer Learning Model Centre"""
import logging
from abc import ABC, abstractmethod
from typing import Callable

import keras
import tensorflow as tf
from keras import layers


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=not-callable, consider-iterating-dictionary
# ------------------------------------------------------------------------------------------------ #
#                                ABSTRACT MODEL FACTORY                                            #
# ------------------------------------------------------------------------------------------------ #
class ModelFactory(ABC):
    """Abstract Model Factory

    Args:
        input_shape (tuple): Shape of input. Default = (224,224,3).
        output_shape (int): Shape of output. Default = 1.
        activation (str): Output activation. Default = 'sigmoid'.
    """

    __model_selection = {
        "DenseNet": {
            "base_model": tf.keras.applications.densenet.DenseNet201,
            "preprocessor": tf.keras.applications.densenet.preprocess_input,
        },
        "EfficientNet": {
            "base_model": tf.keras.applications.efficientnet_v2.EfficientNetV2S,
            "preprocessor": tf.keras.applications.efficientnet.preprocess_input,
        },
        "Inception": {
            "base_model": tf.keras.applications.inception_v3.InceptionV3,
            "preprocessor": tf.keras.applications.inception_v3.preprocess_input,
        },
        "InceptionResNet": {
            "base_model": tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
            "preprocessor": tf.keras.applications.inception_resnet_v2.preprocess_input,
        },
        "MobileNet": {
            "base_model": tf.keras.applications.mobilenet_v2.MobileNetV2,
            "preprocessor": tf.keras.applications.mobilenet_v2.preprocess_input,
        },
        "ResNet": {
            "base_model": tf.keras.applications.resnet_v2.ResNet152V2,
            "preprocessor": tf.keras.applications.resnet_v2.preprocess_input,
        },
        "VGG": {
            "base_model": tf.keras.applications.vgg19.VGG19,
            "preprocessor": tf.keras.applications.vgg19.preprocess_input,
        },
        "Xception": {
            "base_model": tf.keras.applications.xception.Xception,
            "preprocessor": tf.keras.applications.xception.preprocess_input,
        },
    }

    def __init__(
        self,
        version: str,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        output_shape: int = 1,
        activation: str = "sigmoid",
    ) -> None:
        self._version = version
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._activation = activation
        self._input = tf.keras.Input(
            shape=self._input_shape, batch_size=None, name="input_layer"
        )

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def model_names(self) -> list:
        return [k for k in self.__model_selection.keys()]

    @property
    def input(self) -> tf.keras.Input:
        return self._input

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
    def version(self) -> str:
        """Return the model architecture version."""
        return self._version

    @abstractmethod
    def extract_features(
        self, x: tf.Tensor, pretrained_model: Callable, name: str
    ) -> tf.Tensor:
        """Extracts features from feature extraction layers

        Args:
            x (tf.Tensor): TensorFlow Tensor
            pretrained_model (tf.keras.Model): Pretrained model.
            name (str): Name for the model.
        """

    @abstractmethod
    def create_ensemble(self, names: list) -> tf.keras.Model:
        """Creates an ensemble model for the given architecture."""

    def create_model(self, name: str) -> tf.keras.Model:
        """Creates a CNN model based on the designated base model."""
        # Create the feature extraction output for the designated model.
        output = self.create_model_output(name=name)
        # Create final model
        model = tf.keras.Model(self.input, output)
        # Add metadata for ML Ops
        model.alias = name
        model.version = self.version
        return model

    def create_model_output(self, name: str) -> tf.Tensor:
        self._validate_model(name=name)
        # Obtain the pretrained model and freeze all layers.
        pretrained_model = self.__model_selection[name]["base_model"](include_top=False)
        # Explicitly set trainability on each layer so that the flags
        # persist when the model is saved and reloaded.
        pretrained_model.trainable = True
        for layer in pretrained_model.layers:
            layer.trainable = False

        # Perform model specific data preprocessing
        x = self.__model_selection[name]["preprocessor"](self.input)
        # Apply feature extraction layers to pretrained model and return output layer
        return self.extract_features(x=x, pretrained_model=pretrained_model, name=name)

    def _validate_model(self, name: str) -> None:
        if name not in self.__model_selection.keys():
            msg = f"Model {name} does not exist."
            self._logger.exception(msg)
            raise ValueError(msg)

# ------------------------------------------------------------------------------------------------ #
#                                     MODEL INDUSTRY                                               #
# ------------------------------------------------------------------------------------------------ #
class ModelIndustry:
    """Creates concrete factories based upon the model version.

    Args:
        version (str): The version of the factor.
    """
    __factories = {"V1": V1ModelFactory}

    def get_factory(self) ->

# ------------------------------------------------------------------------------------------------ #
#                                   MODEL FACTORY v1                                               #
# ------------------------------------------------------------------------------------------------ #
class V1ModelFactory(ModelFactory):
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

    __version = "V1"

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        output_shape: int = 1,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__(
            version=self.__version,
            input_shape=input_shape,
            output_shape=output_shape,
            activation=activation,
        )
        self._pretrained_model = None

    def extract_features(
        self, x: tf.Tensor, pretrained_model: Callable, name: str
    ) -> tf.Tensor:
        """Extracts features from feature extraction layers

        Args:
            x (tf.Tensor): TensorFlow Tensor
            pretrained_model (tf.keras.Model): Pretrained model.
        """

        # Augment the image data
        x = self.augmentation(x)
        # Feed base model
        x = pretrained_model(x)
        # Pooling for dimensionality reduction
        x = tf.keras.layers.GlobalAveragePooling2D(
            name=f"{name.lower()}_global_average_pooling"
        )(x)

        # Add fully connected layers with dropout for regularization
        x = tf.keras.layers.Dense(
            1024, activation="relu", name=f"{name.lower()}_dense_1"
        )(x)
        x = tf.keras.layers.Dropout(0.5, name=f"{name.lower()}_dropout_1")(x)
        x = tf.keras.layers.Dense(
            1024, activation="relu", name=f"{name.lower()}_dense_2"
        )(x)
        x = tf.keras.layers.Dropout(0.3, name=f"{name.lower()}_dropout_2")(x)
        x = tf.keras.layers.Dense(
            512, activation="relu", name=f"{name.lower()}_dense_3"
        )(x)
        x = tf.keras.layers.Dense(
            128, activation="relu", name=f"{name.lower()}_dense_4"
        )(x)

        # Apply output layer
        outputs = tf.keras.layers.Dense(
            units=self._output_shape,
            activation=self._activation,
            name=f"{name.lower()}_output_layer",
        )(x)

        return outputs

    def create_ensemble(self, names: list) -> tf.keras.Model:
        """Creates an average ensemble model with the designated models

        Args:
            names (list): List of model names for models to include in the ensemble.
        """
        # get output for each model.
        outputs = [self.create_model_output(name) for name in names]

        # Take average of the outputs
        x = tf.keras.layers.Average()(outputs)

        # Create output
        output = tf.keras.layers.Dense(
            units=self._output_shape,
            activation=self._activation,
            name=f"ensemble_{self.version}_output_layer",
        )(x)

        # Create average ensemble model
        model = tf.keras.Model(self.input, output)
        # Add metadata for ML Ops
        model.alias = "Ensemble"
        model.version = self.version

        return model
