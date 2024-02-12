#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/network/shainnet.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 10th 2024 09:56:45 am                                             #
# Modified   : Sunday February 11th 2024 06:52:36 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Model Factory Module"""
from dataclasses import asdict, dataclass

import tensorflow as tf

from bcd.model.base import BaseModel
from bcd.model.network.base import Network, NetworkConfig, NetworkFactory


# ------------------------------------------------------------------------------------------------ #
#                                    SHAIN NET CONFIG                                              #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ShainNetConfig(NetworkConfig):
    """Configuration for ShainNet Network"""

    dense1: int = 1024
    dropout1: float = 0.5
    dense2: int = 1024
    dropout2: float = 0.3
    dense3: int = 512
    dense4: int = 128


# ------------------------------------------------------------------------------------------------ #
#                                     ShaneNetFactory                                              #
# ------------------------------------------------------------------------------------------------ #
class ShainNetFactory(NetworkFactory):
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

    __name = "ShainNet"

    def __init__(
        self,
        config: ShainNetConfig,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        output_shape: int = 1,
        activation: str = "sigmoid",
    ) -> None:
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._config = config
        self._activation = activation

    def create(self, base_model: BaseModel) -> tf.keras.Model:
        """Creates a CNN transfer learning model for the given base model.

        Args:
            base_model (BaseModel): Base model object containing the pretrained model
                and the model specific preprocessor.
        """
        # Designate a model name that will be used to name runs.
        name = f"{self.__name}_{base_model.name}"
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
        x = tf.keras.layers.Dense(
            self._config.dense1, activation="relu", name=f"{name}_dense_1"
        )(x)
        x = tf.keras.layers.Dropout(self._config.dropout1, name=f"{name}_dropout_1")(x)
        x = tf.keras.layers.Dense(
            self._config.dense2, activation="relu", name=f"{name}_dense_2"
        )(x)
        x = tf.keras.layers.Dropout(self._config.dropout2, name=f"{name}_dropout_2")(x)
        x = tf.keras.layers.Dense(
            self._config.dense3, activation="relu", name=f"{name}_dense_3"
        )(x)

        # Add Layers for classification
        x = tf.keras.layers.Dense(
            self._config.dense4, activation="relu", name=f"{name}_dense_4"
        )(x)
        outputs = tf.keras.layers.Dense(
            units=self._output_shape,
            activation=self._activation,
            name=f"{name}_output_layer",
        )(x)

        # Create the model
        model = tf.keras.Model(inputs, outputs)

        # Create the network
        name = self.__name + "_" + base_model.name
        network = Network(
            name=name,
            model=model,
            base_model=base_model.model,
            architecture=self.__name,
            config=self._config,
        )

        return network


# ------------------------------------------------------------------------------------------------ #
#                                       TMNet FActory                                              #
# ------------------------------------------------------------------------------------------------ #
class TMNetFactory(NetworkFactory):
    """Factory for CNN TMNet Transfer Learning model [1]_

    Models are comprised of a frozen pre-trained model upon which, the following layers are added:
    - Global Average Pooling Layer
    - Batch Normalization Layer
    - Dense layer with 1024 nodes and ReLU activation
    - Dense layer with 1024 nodes and ReLU activation
    - Dense layer with sigmoid activation

    Args:
        input_shape (tuple): Shape of input. Default = (224,224,3).
        output_shape (int): Shape of output. Default = 1.
        activation (str): Output activation. Default = 'sigmoid'.

    Reference:
    .. [1] T. Mahmood, J. Li, Y. Pei, and F. Akhtar, “An Automated In-Depth Feature Learning Algorithm
    for Breast Abnormality Prognosis and Robust Characterization from Mammography Images
    Using Deep Transfer Learning,” Biology, vol. 10, no. 9, p. 859, Sep. 2021,
    doi: 10.3390/biology10090859.


    """

    __name = "TMNet"

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
        # Designate a model name that will be used to name runs.
        name = f"{self.__name}_{base_model.name}"
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
        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization(name=f"{name}_batch_normalization")(x)
        # Add fully connected layers
        x = tf.keras.layers.Dense(1024, activation="relu", name=f"{name}_dense_1")(x)
        x = tf.keras.layers.Dense(1024, activation="relu", name=f"{name}_dense_2")(x)

        # Add Layers for classification
        outputs = tf.keras.layers.Dense(
            units=self._output_shape,
            activation=self._activation,
            name=f"{name}_output_layer",
        )(x)
        # Create the model
        model = tf.keras.Model(inputs, outputs)
        # Create the network
        name = self.__name + "_" + base_model.name
        network = Network(
            name=name,
            model=model,
            base_model=base_model.model,
            architecture=self.__name,
        )

        return network


# ------------------------------------------------------------------------------------------------ #
#                                       AKNet FActory                                              #
# ------------------------------------------------------------------------------------------------ #
class AKNetFactory(NetworkFactory):
    """Factory for CNN AKNet Transfer Learning model

    Models are comprised of a frozen pre-trained model upon which, the following layers are added:
    - Global Average Pooling Layer
    - Dense layer with 4096 nodes and ReLU activation
    - Dense layer with 4096 nodes and ReLU activation
    - Dense layer with sigmoid activation

    Args:
        input_shape (tuple): Shape of input. Default = (224,224,3).
        output_shape (int): Shape of output. Default = 1.
        activation (str): Output activation. Default = 'sigmoid'.

    Source:


    """

    __name = "AKNet"

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
        # Designate a model name that will be used to name runs.
        name = f"{self.__name}_{base_model.name}"
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
        # Flatten the output from the base model.
        x = tf.keras.layers.Flatten(name=f"{name}_flatten")(x)
        # Add fully connected layers
        x = tf.keras.layers.Dense(4096, activation="relu", name=f"{name}_dense_1")(x)
        x = tf.keras.layers.Dense(4096, activation="relu", name=f"{name}_dense_2")(x)

        # Add Layers for classification
        outputs = tf.keras.layers.Dense(
            units=self._output_shape,
            activation=self._activation,
            name=f"{name}_output_layer",
        )(x)
        # Create the model
        model = tf.keras.Model(inputs, outputs)
        # Create the network
        name = self.__name + "_" + base_model.name
        network = Network(
            name=name,
            model=model,
            base_model=base_model.model,
            architecture=self.__name,
            register_model=False,
        )

        return network
