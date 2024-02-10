#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/experiment.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday February 6th 2024 12:39:23 am                                               #
# Modified   : Saturday February 10th 2024 10:41:16 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
# pylint: disable=wrong-import-order
# ------------------------------------------------------------------------------------------------ #
import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

from bcd.model.artifact import ModelArtifact
from bcd.model.repo import ModelRepo

# ------------------------------------------------------------------------------------------------ #
load_dotenv()

class Builder(ABC):
    """Abstract base class for model builders."""

    @property
    @abstractmethod
    def experiment(self) -> tf.keras.Model:
        """Returns the model"""

    @abstractmethod
    def build_feature_extraction(self, name: str, )

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
class Experiment:
    """Encapsulates a single experiment or run of an experimental model."""

    def __init__(
        self,
        model: tf.keras.Model,
        config: dict,
        optimizer: tf.keras.optimizers,
        repo: ModelRepo,
        callbacks: list = None,
        metrics: list = None,
        force: bool = False,
    ) -> None:
        self._model = model
        self._config = config
        self._optimizer = optimizer
        self._repo = repo
        self._callbacks = callbacks
        self._metrics = metrics
        self._force = force

        self._history = None
        self._run = None
        self._model_artifact = ModelArtifact(
            name=model.alias, version=model.version, dataset=self._config["dataset"]
        )
        self._entity = os.getenv("WANDB_ENTITY")

        self._logger = logging.getLogger(
            f"{self.__class__.__name__}-{self._model_artifact.id}"
        )

    @property
    def model_artifact_id(self) -> str:
        return self._model_artifact.id

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def history(self) -> tf.keras.callbacks.History:
        return self._history

    def run(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
    ) -> tf.keras.Model:
        """Compiles and fits the model.

        Args:
            train_ds (tf.data.Dataset): Training Dataset
            val_ds (tf.data.Dataset): Validation Dataset

        """
        if self._repo.exists(model_id=self._model_artifact.id) and not self._force:
            self._model = self._repo.get(model_id=self._model_artifact.id)
        else:
            # Remove existing model if it exists
            self._repo.remove(model_id=self._model_artifact.id, ignore_errors=True)
            # Remove existing run(s) for the experiment.
            self.remove_existing_runs()

            # Instantiate a wandb run and callback
            self._run = wandb.init(
                project=self._config["project"],
                name=self._config["run_name"],
                config=self._config,
            )
            wandb_callback = wandb.keras.WandbMetricsLogger()
            self._callbacks.append(wandb_callback)

            # For memory and performance efficiency, release the
            # global state keras maintains to implement the
            # Functional API.
            tf.keras.backend.clear_session()

            # Compile the model
            self._model.compile(
                loss=self._config["loss"],
                optimizer=self._optimizer,
                metrics=self._metrics,
            )

            # Fit the model
            self._history = self._model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self._config["epochs"],
                callbacks=self._callbacks,
            )
            # Save the model in the repository
            self._repo.add(model_id=self._model_artifact.id, model=self._model)

            # Register the model as an artifact on wandb
            self._register_model()

            # Finish the run
            wandb.finish()

    def _register_model(self) -> None:
        """Registers the model as an artifact on wandb"""
        filepath = self._repo.get_filepath(model_id=self._model_artifact.id)
        # Upload the model to the wandb model registry
        self._run.log_model(path=filepath, name=self._model_artifact.id)
        # Link the model to the run
        self._run.link_model(
            path=filepath, registered_model_name=self._model_artifact.id
        )

    def remove_existing_runs(self) -> None:
        try:
            runs = wandb.Api().runs(f"{self._entity}/{self._config['project']}")
            for run in runs:
                if run.name == self._config["run_name"]:
                    run.delete()
        except ValueError:
            msg = f"No existing runs were found for project: {self._config['project']}"
            self._logger.info(msg)
