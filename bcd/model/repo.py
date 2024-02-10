#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filemodel_id   : /bcd/model/repo.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 15th 2024 04:04:13 pm                                                #
# Modified   : Thursday February 8th 2024 11:30:00 pm                                              #
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

    def get(self, model_id: str) -> tf.keras.Model:
        """Returns the model for the designated model_id and stage.

        Args:
            model_id (str): Model model_id
        """
        filepath = self.get_filepath(model_id=model_id)
        try:
            model = tf.keras.models.load_model(filepath)
        except OSError as exc:
            msg = f"Model {model_id} does not exist."
            self._logger.exception(msg=msg)
            raise FileNotFoundError from exc

        msg = f"Loaded {model_id} model from the repository."
        self._logger.info(msg)

        return model

    def add(self, model_id: str, model: tf.keras.Model) -> None:
        """Adds a model to the repository

        Args:
            model_id (str): The model model_id
            model (tf.keras.Model): TensorFlow Model.

        """
        filepath = self.get_filepath(model_id=model_id)

        if not os.path.exists(filepath):
            model.save(filepath)
        else:
            msg = f"Model {model_id} already exists and can't be added."
            self._logger.exception(msg)
            raise FileExistsError(msg)

    def get_filepath(self, model_id: str) -> str:
        """Returns the filepath for the designated model.

        Args:
            model_id (str): Model model_id
        """
        filemodel_id = model_id + ".keras"
        return os.path.join(self.__location, filemodel_id)

    def exists(self, model_id: str) -> bool:
        """Determines whether models exist for the designated model_id and stage

        Args:
            model_id (str): Model model_id
        """
        filepath = self.get_filepath(model_id=model_id)
        return os.path.exists(filepath)

    def remove(self, model_id: str, ignore_errors: bool = True) -> bool:
        """Removes model from the repository

        Args:
            model_id (str): Model model_id
        """
        filepath = self.get_filepath(model_id=model_id)
        try:
            os.remove(filepath)
        except FileNotFoundError:
            if not ignore_errors:
                msg = f"Model {model_id} does not exist."
                self._logger.info(msg)
                raise
            else:
                pass
