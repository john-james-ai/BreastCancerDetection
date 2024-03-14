#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/store.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 15th 2024 04:04:13 pm                                                #
# Modified   : Wednesday February 21st 2024 09:39:38 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
# pylint: disable=wrong-import-order, broad-exception-caught
# ------------------------------------------------------------------------------------------------ #
import logging
import os

import pandas as pd
import tensorflow as tf
import wandb
from dotenv import load_dotenv

from bcd.model.adapter import Adapter

load_dotenv()


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=protected-access
# ------------------------------------------------------------------------------------------------ #
#                                     MODEL REPO                                                   #
# ------------------------------------------------------------------------------------------------ #
class ExperimentRepo:
    """Repository for fitted models."""

    def __init__(self, mode: str, project: str, adapter: Adapter) -> None:
        self._mode = mode
        self._project = project
        self._adapter = adapter
        self._entity = self._adapter.wandb_entity
        self._directory = os.path.join(self._adapter.model_dir, mode.lower())

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def get(self, name: str, alias: str = "latest") -> tf.keras.Model:
        """Returns a named model from Weights & Biases

        Args:
            name (str): The registered model name
            alias (str): String that uniquely identifies a model version. It can be
                one automatically assigned by W%B such as 'latest' or 'v0'.

        """
        run = wandb.init()
        model_name = f"{self._entity}/{self._project}/{name}:{alias}"
        try:
            filepath = run.use_model(model_name)
            return tf.keras.models.load_model(filepath)
        except Exception as exc:
            msg = f"Model {name}:{alias} not found."
            self._logger.exception(msg)
            raise FileNotFoundError(msg) from exc

    def add(self, run: wandb.run, name: str, filepath: str) -> None:
        """Registers the model as an artifact on wandb

        Args:
            run (wandb.run): A Weights & Biases run object.
            name (str): The name of the model.
            filepath (str): The local filepath for the model.
        """
        # Upload the model to the wandb model registry
        run.log_model(path=filepath, name=name)
        # Link the model to the run
        run.link_model(path=filepath, registered_model_name=name)

    def load(self, name: str, model_id: str) -> tf.keras.Model:
        """Reads a model from file.

        Args:
            name (str): Model name
            model_id (str): The id used to uniquely distinguish the model
        """
        filepath = self.get_filepath(name=name, model_id=model_id)
        try:
            return tf.keras.models.load_model(filepath)
        except OSError as exc:
            msg = f"Model {name} not found."
            self._logger.exception(msg=msg)
            raise FileNotFoundError(msg) from exc

    def save(
        self, model_id: str, name: str, model: tf.keras.Model, force: bool = True
    ) -> None:
        """Adds a model to the repository

        Args:
            model_id (str): A Weights and Bias Run id.
            name (str): The model name
            model (tf.keras.Model): TensorFlow Model.

        """
        filepath = self.get_filepath(name=name, model_id=model_id)

        if not os.path.exists(filepath) or force:
            model.save(filepath)
        else:
            msg = f"Model {name} already exists and can't be added."
            self._logger.exception(msg)
            raise FileExistsError(msg)
        return filepath

    def get_filepath(self, name: str, model_id: str, weights_only: bool = False) -> str:
        """Returns the filepath for the designated model and model_id .

        Args:
            name (str): Model name
            model_id (str): The id used to uniquely distinguish the model
        """
        if weights_only:
            filename = f"{name}-{model_id}.weights.h5"
        else:
            filename = f"{name}-{model_id}.keras"
        return os.path.join(self._directory, filename)

    def exists(self, name: str, config_hash: str) -> bool:
        """Checks existence of an experiment with the same configuration

        Args:
            name (str): Model name
            config_hash (str): The configuration hash
        """
        try:
            runs = wandb.Api().runs(f"{self._entity}/{self._project}")
            for run in runs:
                try:
                    if run.config["hash"] == config_hash and run.name == name:
                        return True
                except KeyError:
                    pass
            return False
        except ValueError:
            return False

    def export_runs(self, filepath: str) -> None:

        api = wandb.Api()

        # Project is specified by <entity/project-name>
        runs = api.runs("aistudio/Breast-Cancer-Detection-Production")

        summary_list = []
        for run in runs:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)

        runs_df = pd.DataFrame(summary_list)

        runs_df.to_csv(filepath)
