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
# Created    : Friday January 12th 2024 08:35:25 pm                                                #
# Modified   : Sunday January 21st 2024 11:16:35 pm                                                #
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
#                                   MODEL FACTORY                                                  #
# ------------------------------------------------------------------------------------------------ #
class ModelFactory(ABC):
    """Encapsulates model creation."""

    @property
    @abstractmethod
    def base_model(self) -> tf.keras.Model:
        """Factory method that returns the base model. Behavior defined in subclasses."""

    @property
    def augmentation(self) -> tf.Tensor:
        """Performs random augmentation on images.

        Args:
            inputs (tf.Tensor): Input data
        """
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.2),
                layers.RandomHeight(0.2),
                layers.RandomWidth(0.2),
            ],
            name="data_augmentation",
        )

    def create(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        output_shape: int = 1,
        activation: str = "sigmoid",
        trainable: bool = False,
    ) -> tf.keras.Model:
        """Creates the model with built-in augmentation.

        Args:
            input_shape (tuple): Expected shape of the input images. Default is (224,224,3).
            output_shape (int): Number of classes for the output layer.  Default is 1.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            trainable (bool): Whether the base model is trainable. Default is False.
            loss (str): The loss function. Default is 'binary_crossentropy'.
            activation (str): The activation function. Default is "sigmoid".
            metrics (list): Metrics to track.

        Returns: Compiled TensorFlow Keras Model
        """

        # Create the input
        inputs = tf.keras.Input(shape=input_shape, batch_size=None, name="input_layer")
        # Perform base model specific preprocessing
        x = self.preprocess_input(inputs=inputs)
        # Add augmentation model as a layer
        x = self.augmentation(x)
        # Give base model inputs (after augmentation) in inference model
        x = self.base_model(x, training=trainable)
        # Pool output features of base model
        x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
        # Add output layer
        outputs = layers.Dense(
            units=output_shape, activation=activation, name="output_layer"
        )(x)
        # Create the model with inputs and outputs
        model = keras.Model(inputs, outputs)

        return model

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs preprocessing on inputs. Model-specific processing overridden in subclasses.

        Args:
            inputs (tf.Tensor): Model inputs
        """
        return inputs


# ------------------------------------------------------------------------------------------------ #
#                                      DENSENET                                                    #
# ------------------------------------------------------------------------------------------------ #
class DenseNetFactory(ModelFactory):
    """Instantiates the DenseNet201 Architecture"""

    @property
    def base_model(self) -> tf.keras.Model:
        model = tf.keras.applications.densenet.DenseNet201(include_top=False)
        model.trainable = False
        return model

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.densenet.preprocess_input(inputs)


# ------------------------------------------------------------------------------------------------ #
#                                  EFFICIENT NET                                                   #
# ------------------------------------------------------------------------------------------------ #
class EfficientNetFactory(ModelFactory):
    """Instantiates the Efficient Net Architecture"""

    @property
    def base_model(self) -> tf.keras.Model:
        model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False)
        model.trainable = False
        return model


# ------------------------------------------------------------------------------------------------ #
#                                      INCEPTION                                                   #
# ------------------------------------------------------------------------------------------------ #
class InceptionFactory(ModelFactory):
    """Instantiates the Inception v3 architecture."""

    @property
    def base_model(self) -> tf.keras.Model:
        model = tf.keras.applications.inception_v3.InceptionV3(include_top=False)
        model.trainable = False
        return model

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.inception_v3.preprocess_input(inputs)


# ------------------------------------------------------------------------------------------------ #
#                                  INCEPTION RESNET                                                #
# ------------------------------------------------------------------------------------------------ #
class InceptionResNetFactory(ModelFactory):
    """Instantiates the Inception-ResNet v2 Architecture"""

    @property
    def base_model(self) -> tf.keras.Model:
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False
        )
        model.trainable = False
        return model

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.inception_resnet_v2.preprocess_input(inputs)


# ------------------------------------------------------------------------------------------------ #
#                                       MOBILENET                                                  #
# ------------------------------------------------------------------------------------------------ #
class MobileNetFactory(ModelFactory):
    """Instantiates the MobileNetV2 Architecture"""

    @property
    def base_model(self) -> tf.keras.Model:
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False)
        model.trainable = False
        return model

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)


# ------------------------------------------------------------------------------------------------ #
#                                     RESNET                                                       #
# ------------------------------------------------------------------------------------------------ #
class ResNetFactory(ModelFactory):
    """Instantiates the ResNet152v2 Architecture"""

    @property
    def base_model(self) -> tf.keras.Model:
        model = tf.keras.applications.resnet_v2.ResNet152V2(include_top=False)
        model.trainable = False
        return model

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.resnet_v2.preprocess_input(inputs)


# ------------------------------------------------------------------------------------------------ #
#                                       VGG                                                        #
# ------------------------------------------------------------------------------------------------ #
class VGGFactory(ModelFactory):
    """Instantiates the VGG19 Architecture"""

    @property
    def base_model(self) -> tf.keras.Model:
        model = tf.keras.applications.vgg19.VGG19(include_top=False)
        model.trainable = False
        return model

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.vgg19.preprocess_input(inputs)


# ------------------------------------------------------------------------------------------------ #
#                                       XCEPTION                                                   #
# ------------------------------------------------------------------------------------------------ #
class XceptionFactory(ModelFactory):
    """Instantiates the Xception architecture."""

    @property
    def base_model(self) -> tf.keras.Model:
        model = tf.keras.applications.xception.Xception(include_top=False)
        model.trainable = False
        return model

    def preprocess_input(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.xception.preprocess_input(inputs)
