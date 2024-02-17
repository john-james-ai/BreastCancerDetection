#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/adapter.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday February 15th 2024 08:48:53 pm                                             #
# Modified   : Saturday February 17th 2024 10:35:06 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import os
from abc import ABC, abstractmethod

import tensorflow as tf

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=import-outside-toplevel, import-error
# ------------------------------------------------------------------------------------------------ #
DATASETS = {
    "Development": {
        "name": "CBIS-DDSM_10",
        "directory": "data/image/1_final/training_10/training/",
    },
    "Stage": {
        "name": "CBIS-DDSM_30",
        "directory": "data/image/1_final/training_30/training/",
    },
    "Production": {
        "name": "CBIS-DDSM",
        "directory": "data/image/1_final/training/training/",
    },
}


# ------------------------------------------------------------------------------------------------ #
class Adapter(ABC):
    """Defines interface for Adapter subclasses."""

    @property
    def tf_version(self) -> str:
        """Returns the TensorFlow version"""
        return tf.__version__

    @property
    @abstractmethod
    def device_type(self) -> str:
        """Return device type as 'CPU', 'GPU', or 'TPU'."""

    @property
    @abstractmethod
    def wandb_api_key(self) -> str:
        """Returns the Weights & Biases API Key"""

    @property
    @abstractmethod
    def wandb_entity(self) -> str:
        """Returns the Weights & Biases API Key"""

    @property
    @abstractmethod
    def train_dir(self) -> str:
        """Returns the base directory for the training set."""

    @property
    @abstractmethod
    def model_dir(self) -> str:
        """Returns the model directory."""

    def get_strategy(self) -> tf.distribute.Strategy:
        """Returns the TensorFlow compute strategy."""
        # Detect hardware, return appropriate distribution strategy

        try:
            # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        except ValueError:
            tpu = None

        if tpu:
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        else:
            strategy = (
                tf.distribute.get_strategy()
            )  # default distribution strategy in Tensorflow. Works on CPU and single GPU.

        return strategy


# ------------------------------------------------------------------------------------------------ #
#                                            LOCAL                                                 #
# ------------------------------------------------------------------------------------------------ #
class LocalAdapter(Adapter):
    """Encapsulates variables specific to the local environment.

    Args:
        mode (str): Either 'Development', 'Stage',  or 'Production'.
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode

    @property
    def device_type(self) -> str:
        """Return device type as 'CPU', 'GPU', or 'TPU'."""
        return "CPU"

    @property
    def wandb_api_key(self) -> str:
        """Returns the Weights & Biases API Key"""
        from dotenv import load_dotenv

        load_dotenv()
        return os.getenv("WANDB_API_KEY")

    @property
    def wandb_entity(self) -> str:
        """Returns the Weights & Biases API Key"""
        from dotenv import load_dotenv

        load_dotenv()
        return os.getenv("WANDB_ENTITY")

    @property
    def train_dir(self) -> str:
        """Returns the base directory for the training set."""
        try:
            return DATASETS[self._mode]["directory"]
        except KeyError:
            msg = f"Invalid mode. Valid modes are {DATASETS.keys()}"
            logging.exception(msg)
            raise

    @property
    def model_dir(self) -> str:
        """Returns the model checkpoint directory."""
        return "models/"


# ------------------------------------------------------------------------------------------------ #
#                                           KAGGLE                                                 #
# ------------------------------------------------------------------------------------------------ #
class KaggleAdapter(Adapter):
    """Encapsulates variables specific to the local environment.

    Args:
        mode (str): Either 'Development', 'Stage',  or 'Production'.
    """

    __train_dir = "cbis-ddsm-training-set"

    def __init__(self, mode: str) -> None:
        self._mode = mode

    @property
    def device_type(self) -> str:
        """Return device type as 'CPU', 'GPU', or 'TPU'."""
        device_name = tf.test.gpu_device_name()
        if "GPU" in device_name:
            return "GPU"
        else:
            try:
                _ = tf.distribute.cluster_resolver.TPUClusterResolver()
                return "TPU"
            except ValueError:
                return "CPU"

    @property
    def wandb_api_key(self) -> str:
        """Returns the Weights & Biases API Key"""
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        return user_secrets.get_secret("WANDB_API_KEY")

    @property
    def wandb_entity(self) -> str:
        """Returns the Weights & Biases entity"""
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        return user_secrets.get_secret("WANDB_ENTITY")

    @property
    def train_dir(self) -> str:
        """Returns the base directory for the training set."""
        return f"/kaggle/input/{self.__train_dir}"

    @property
    def model_dir(self) -> str:
        """Returns the model directory."""
        return "/kaggle/working/models/"
