#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/repo.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 15th 2024 04:04:13 pm                                                #
# Modified   : Wednesday February 7th 2024 07:00:23 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import os

import tensorflow as tf


# ------------------------------------------------------------------------------------------------ #
class ModelRepo:
    """Repository for fitted models."""

    __location = "models/"

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(level=logging.DEBUG)

    def get(self, name: str) -> tf.keras.Model:
        """Returns the model for the designated name and stage.

        Args:
            name (str): Model name
        """
        filepath = self.get_filepath(name=name)
        try:
            model = tf.keras.models.load_model(filepath)
        except OSError as exc:
            msg = f"Model {name} does not exist."
            self._logger.exception(msg=msg)
            raise FileNotFoundError from exc

        msg = f"Loaded {name} model from the repository."
        self._logger.info(msg)

        return model

    def add(self, name: str, model: tf.keras.Model) -> None:
        """Adds a model to the repository

        Args:
            name (str): The model name
            model (tf.keras.Model): TensorFlow Model.

        """
        filepath = self.get_filepath(name=name)

        if not os.path.exists(filepath):
            model.save(filepath)
        else:
            msg = f"Model {name} already exists and can't be added."
            self._logger.exception(msg)
            raise FileExistsError(msg)

    def get_filepath(self, name: str) -> str:
        """Returns the filepath for the designated model.

        Args:
            name (str): Model name
        """
        filename = name + ".keras"
        return os.path.join(self.__location, filename)

    def exists(self, name: str) -> bool:
        """Determines whether models exist for the designated name and stage

        Args:
            name (str): Model name
        """
        filepath = self.get_filepath(name=name)
        return os.path.exists(filepath)

    def remove(self, name: str, ignore_errors: bool = True) -> bool:
        """Removes model from the repository

        Args:
            name (str): Model name
        """
        filepath = self.get_filepath(name=name)
        try:
            os.remove(filepath)
        except FileNotFoundError:
            if not ignore_errors:
                msg = f"Model {name} does not exist."
                self._logger.info(msg)
                raise
            else:
                pass
