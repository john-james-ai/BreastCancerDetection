#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/base.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday February 9th 2024 01:39:56 am                                                #
# Modified   : Saturday February 10th 2024 09:54:58 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Pretrained Model Module"""
from abc import ABC, abstractmethod
from typing import Callable

import tensorflow as tf


# ------------------------------------------------------------------------------------------------ #
class BaseModel(ABC):
    """Abstract pretrained class for pre-trained CNN pretrained models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name for the pretrained model."""

    @property
    @abstractmethod
    def model(self) -> tf.keras.Model:
        """Returns the pretrained model."""

    @property
    @abstractmethod
    def preprocessor(self) -> Callable:
        """Returns the model specific data preprocessor"""


# ------------------------------------------------------------------------------------------------ #
#                                         DenseNet                                                 #
# ------------------------------------------------------------------------------------------------ #
class DenseNet(BaseModel):
    """Encapsulates the DenseNet model and input preprocessing."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def model(self) -> tf.keras.Model:
        """Instantiates and returns the feature extraction network."""
        model = tf.keras.applications.densenet.DenseNet201(include_top=False)
        model.trainable = True
        for layer in model.layers:
            layer.trainable = False
        return model

    @property
    def preprocessor(self) -> Callable:
        """Returns the model specific data preprocessor"""
        return tf.keras.applications.densenet.preprocess_input


# ------------------------------------------------------------------------------------------------ #
#                                         EfficientNet                                             #
# ------------------------------------------------------------------------------------------------ #
class EfficientNet(BaseModel):
    """Encapsulates the EfficientNet model and input preprocessing."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def model(self) -> tf.keras.Model:
        """Instantiates and returns the feature extraction network."""
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False)
        model.trainable = True
        for layer in model.layers:
            layer.trainable = False
        return model

    @property
    def preprocessor(self) -> Callable:
        """Returns the model specific data preprocessor"""
        return tf.keras.applications.efficientnet.preprocess_input


# ------------------------------------------------------------------------------------------------ #
#                                         Inception                                                #
# ------------------------------------------------------------------------------------------------ #
class Inception(BaseModel):
    """Encapsulates the Inception model and input preprocessing."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def model(self) -> tf.keras.Model:
        """Instantiates and returns the feature extraction network."""
        model = tf.keras.applications.inception_v3.InceptionV3(include_top=False)
        model.trainable = True
        for layer in model.layers:
            layer.trainable = False
        return model

    @property
    def preprocessor(self) -> Callable:
        """Returns the model specific data preprocessor"""
        return tf.keras.applications.inception_v3.preprocess_input


# ------------------------------------------------------------------------------------------------ #
#                                        InceptionResNet                                           #
# ------------------------------------------------------------------------------------------------ #
class InceptionResNet(BaseModel):
    """Encapsulates the InceptionResNet model and input preprocessing."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def model(self) -> tf.keras.Model:
        """Instantiates and returns the feature extraction network."""
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False
        )
        model.trainable = True
        for layer in model.layers:
            layer.trainable = False
        return model

    @property
    def preprocessor(self) -> Callable:
        """Returns the model specific data preprocessor"""
        return tf.keras.applications.inception_resnet_v2.preprocess_input


# ------------------------------------------------------------------------------------------------ #
#                                         MobileNet                                                #
# ------------------------------------------------------------------------------------------------ #
class MobileNet(BaseModel):
    """Encapsulates the MobileNet model and input preprocessing."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def model(self) -> tf.keras.Model:
        """Instantiates and returns the feature extraction network."""
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False)
        model.trainable = True
        for layer in model.layers:
            layer.trainable = False
        return model

    @property
    def preprocessor(self) -> Callable:
        """Returns the model specific data preprocessor"""
        return tf.keras.applications.mobilenet_v2.preprocess_input


# ------------------------------------------------------------------------------------------------ #
#                                         ResNet                                                   #
# ------------------------------------------------------------------------------------------------ #
class ResNet(BaseModel):
    """Encapsulates the ResNet model and input preprocessing."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def model(self) -> tf.keras.Model:
        """Instantiates and returns the feature extraction network."""
        model = tf.keras.applications.resnet_v2.ResNet152V2(include_top=False)
        model.trainable = True
        for layer in model.layers:
            layer.trainable = False
        return model

    @property
    def preprocessor(self) -> Callable:
        """Returns the model specific data preprocessor"""
        return tf.keras.applications.resnet_v2.preprocess_input


# ------------------------------------------------------------------------------------------------ #
#                                           VGG                                                    #
# ------------------------------------------------------------------------------------------------ #
class VGG(BaseModel):
    """Encapsulates the VGG model and input preprocessing."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def model(self) -> tf.keras.Model:
        """Instantiates and returns the feature extraction network."""
        model = tf.keras.applications.vgg19.VGG19(include_top=False)
        model.trainable = True
        for layer in model.layers:
            layer.trainable = False
        return model

    @property
    def preprocessor(self) -> Callable:
        """Returns the model specific data preprocessor"""
        return tf.keras.applications.vgg19.preprocess_input


# ------------------------------------------------------------------------------------------------ #
#                                         Xception                                                 #
# ------------------------------------------------------------------------------------------------ #
class Xception(BaseModel):
    """Encapsulates the Xception model and input preprocessing."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def model(self) -> tf.keras.Model:
        """Instantiates and returns the feature extraction network."""
        model = tf.keras.applications.xception.Xception(include_top=False)
        model.trainable = True
        for layer in model.layers:
            layer.trainable = False
        return model

    @property
    def preprocessor(self) -> Callable:
        """Returns the model specific data preprocessor"""
        return tf.keras.applications.xception.preprocess_input
