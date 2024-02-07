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
# Modified   : Wednesday February 7th 2024 06:57:11 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
# pylint: disable=wrong-import-order
# ------------------------------------------------------------------------------------------------ #
import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import classification_report, confusion_matrix

from bcd.model.repo import ModelRepo


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
        self._name = model.alias + "_" + model.version + "-" + self._config["dataset"]

        self._logger = logging.getLogger(f"{self.__class__.__name__}-{self._name}")

    @property
    def name(self) -> str:
        return self._name

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
        if self._repo.exists(name=self._name) and not self._force:
            self._model = self._repo.get(name=self._name)
        else:
            # Remove existing model if it exists
            self._repo.remove(name=self._name, ignore_errors=True)

            # Instantiate a wandb run and callback
            self._run = wandb.init(project=self._config["project"], config=self._config)
            wandb_callback = wandb.keras.WandbMetricsLogger()
            self._callbacks.append(wandb_callback)

            # Clear Keras session prior to compile This may resolve
            # Model save errors.
            # https://stackoverflow.com/questions/72776335/valueerror-unable-to-create-dataset-name-already-exists-when-using-modelcheck
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
            self._repo.add(name=self._name, model=self._model)

            # Register the model as an artifact on wandb
            self._register_model()

            # Finish the run
            wandb.finish()

    def evaluate(self, data: tf.data.Dataset) -> dict:
        return self._model.evaluate(data)

    def predict(self, data: tf.data.Dataset) -> np.ndarray:
        return (self._model.predict(data) > 0.5).astype("int32")

    def classification_report(self, data: tf.data.Dataset) -> None:
        actual = np.concatenate([y for x, y in data], axis=0)
        predicted = self.predict(data=data)
        print(classification_report(actual, predicted, target_names=data.class_names))

    def plot_confusion_matrix(self, data: tf.data.Dataset) -> None:
        """Plots a confusion matrix for the validation set.

        Args:
            data (tf.data.Dataset): Dataset for which the confusion matrix is
                to be computed.

        """
        actual = np.concatenate([y for x, y in data], axis=0)
        predicted = self.predict(data=data)

        cm = confusion_matrix(actual, predicted)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(4, 4))
        cmap = "Blues"
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(f"{self._name}\nConfusion matrix", fontsize=12)

        tick_marks = np.arange(len(data.class_names))
        plt.xticks(tick_marks, data.class_names, rotation=90, fontsize=10)
        plt.yticks(tick_marks, data.class_names, fontsize=10)

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], ".2f"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

        plt.ylabel("True label", fontsize=10)
        plt.xlabel("Predicted label", fontsize=10)
        plt.show()

    def _register_model(self) -> None:
        """Registers the model as an artifact on wandb"""
        filepath = self._repo.get_filepath(name=self._name)
        artifact = wandb.Artifact(f"{self._name}-{self._run.id}", type="model")
        artifact.add_file(filepath)
        wandb.log_artifact(artifact, aliases=[self._name, "best"])
        artifact_path = "aistudio/" + self._config["project"] + "/" + self._name
        wandb.run.link_artifact(artifact, artifact_path)
