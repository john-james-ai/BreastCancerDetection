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
# Modified   : Monday February 12th 2024 12:16:31 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
# pylint: disable=wrong-import-order, broad-exception-caught
# ------------------------------------------------------------------------------------------------ #
import logging
import os
from glob import glob

import tensorflow as tf
import wandb
from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------------------------------------------------------------ #
#                                     MODEL REPO                                                   #
# ------------------------------------------------------------------------------------------------ #
class ModelRepo:
    """Repository for fitted models."""

    __location = "models/"

    def __init__(self, mode: str, project: str) -> None:
        self._mode = mode
        self._project = project
        self._entity = os.getenv("WANDB_ENTITY")
        self._directory = os.path.join(self.__location, mode.lower())

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def get(self, name: str, alias: str = "latest") -> tf.keras.Model:
        """Returns a named model.

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

    def add(self, run: wandb.run, name: str, model: tf.keras.Model) -> None:
        filepath = self.save(run=run, name=name, model=model)
        self._register(run=run, name=name, filepath=filepath)

    def load(self, name: str, run_id: str) -> tf.keras.Model:
        """Reads a model from file.

        Args:
            name (str): Model name
            run_id (str): Id for the run that created the model.
        """
        filepath = self.get_filepath(name=name, run_id=run_id)
        try:
            return tf.keras.models.load_model(filepath)
        except OSError as exc:
            msg = f"Model {name} not found."
            self._logger.exception(msg=msg)
            raise FileNotFoundError(msg) from exc

    def save(
        self, run: wandb.run, name: str, model: tf.keras.Model, force: bool = True
    ) -> None:
        """Adds a model to the repository

        Args:
            run (str): A Weights and Bias Run object.
            name (str): The model name
            model (tf.keras.Model): TensorFlow Model.

        """
        filepath = self.get_filepath(name=name, run_id=run.id)

        if not os.path.exists(filepath) or force:
            model.save(filepath)
        else:
            msg = f"Model {name} already exists and can't be added."
            self._logger.exception(msg)
            raise FileExistsError(msg)
        return filepath

    def get_filepath(self, name: str, run_id: str) -> str:
        """Returns the filepath for the designated model and run_id .

        Args:
            name (str): Model name
            run_id (str): The run id for the run producing the model.
        """
        filename = f"{name}-{run_id}.keras"
        return os.path.join(self._directory, filename)

    def exists(self, name: str, config: dict = None, run_id: str = None) -> bool:
        """Checks existence of a model.

        Of config is not None, the method checks remotely for a model
        with the same configuration. If config is None, run_id must be provide.
        In which case, the method checks locally for a model that matches
        the name and run_id.

        Args:
            name (str): Model name
            config (dict): Model configuration. Optional. If provided, the
                method checks remotely for existence.
            run_id (str): The run_id associated with the model. If provided,
                the method looks locally for the model.

        Raises:
            ValueError if config and run_id are both None.
        """
        if config is not None:
            return self.experiment_exists(name=name, config=config)
        elif run_id is not None:
            return self.exists_local(name=name, run_id=run_id)
        else:
            msg = "Must provide config or run_id when checking existence of a model."
            self._logger.exception(msg)
            raise ValueError(msg)

    def exists_local(self, name: str, run_id: str) -> bool:
        """Determines whether models exists locally.

        Args:
            name (str): Model name
            run_id (str): Run id that produced the model.
        """
        filepath = self.get_filepath(name=name, run_id=run_id)
        return os.path.exists(filepath)

    def experiment_exists(self, name: str, config: dict) -> bool:
        """Checks existence of a named experiment (run) with the same configuration.

        Args:
            project (str): Project name on Weights and Biases
            name (str): Name or model_name corresponding to the run name
            config (dict): Experiment configuration.

        """
        runs = wandb.Api().runs(f"{self._entity}/{self._project}")
        for run in runs:
            if run.config["hash"] == config["hash"] and run.name == name:
                return True
        return False

    def remove(self, name: str) -> None:
        """Removes all local and remote runs, models and artifacts associated with the named model.

        Args:
            name (str): Model name
        """
        self._remove_local_files(name=name)
        self._remove_remote_files(name=name)

    def _remove_local_files(self, name: str) -> None:
        """Removes all files matching the name."""
        # Remove Local Files
        pattern = f"{self._directory}/{name}*.keras"
        filepaths = glob(pattern, recursive=True)
        if len(filepaths) > 0:
            # Confirm delete
            go = input(f"{len(filepaths)} files will be deleted. Confirm. [Y/N]")

            if "y" in go.lower():
                for filepath in filepaths:
                    os.remove(filepath)
                msg = f"{len(filepaths)} files have been deleted."
                self._logger.info(msg)

    def _remove_remote_files(self, name: str) -> None:
        """Removes remote runs, and model artifacts."""
        n_runs = 0
        n_artifacts = 0

        runs = wandb.Api().runs(f"{self._entity}/{self._project}")
        if len(runs) > 0:
            # Confirm delete
            go = input(f"{len(runs)} runs will be deleted. Confirm. [Y/N]")
            if "y" in go.lower():
                for run in runs:
                    for artifact in run.logged_artifacts():
                        if artifact.type == "model" and run.name == name:
                            n_artifacts += 1
                            artifact.delete(delete_aliases=True)
                            run.delete()
                msg = f"{n_runs} runs and {n_artifacts} artifacts have been deleted."
                self._logger.info(msg)

    def _register(self, run: wandb.run, name: str, filepath: str) -> None:
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
