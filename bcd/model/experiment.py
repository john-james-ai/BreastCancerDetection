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
# Created    : Monday February 12th 2024 04:02:57 pm                                               #
# Modified   : Saturday February 24th 2024 01:24:19 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import sys
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import wandb
from dotenv import load_dotenv

from bcd.model.config import Config
from bcd.model.network.base import Network
from bcd.model.store import ExperimentRepo

# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)


# ------------------------------------------------------------------------------------------------ #
#                                     EXPERIMENT                                                   #
# ------------------------------------------------------------------------------------------------ #
class BaseExperiment(ABC):
    """Abstract base class for experiments."""

    @property
    @abstractmethod
    def run_id(self) -> str:
        """Returns the Weights and Biases run id"""

    @property
    @abstractmethod
    def filepath(self) -> str:
        """Returns filepath for the best saved model"""

    @abstractmethod
    def run(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
    ) -> tf.keras.Model:
        """Runs the experiment."""

    def announce(self, network: Network) -> None:
        print(
            "# ================================================================================================ #"
        )
        name = network.name.center(100)
        print(name)
        print(
            "# ------------------------------------------------------------------------------------------------ #"
        )
        network.summary()
        print(
            "# ------------------------------------------------------------------------------------------------ #"
        )
        print(network.config)
        print(
            "# ================================================================================================ #"
        )

    def create_plots(self, model: tf.keras.Model, data: tf.data.Dataset) -> None:
        """Creates and loads plots to Weights and Biases"""
        actual = np.concatenate([y for x, y in data], axis=0)
        # Probabilities of the positive class.
        probabilities = model.predict(data)
        # Create probabilities for both classes.
        ones = np.ones(probabilities.shape)
        probabilities_2d = np.stack((probabilities, ones - probabilities), axis=1)
        # Predictions
        predicted = (probabilities > 0.5).astype("int32")

        # Plot Confusion Matrix
        cm = wandb.sklearn.plot_confusion_matrix(actual, predicted, data.class_names)
        wandb.log({"confusion_matrix": cm})

        # Plot ROC Curve
        wandb.sklearn.plot_roc(
            y_true=actual, y_probas=probabilities_2d, labels=data.class_names
        )

        # Precision Recall Curve
        wandb.sklearn.plot_precision_recall(
            y_true=actual, y_probas=probabilities_2d, labels=data.class_names
        )


# ------------------------------------------------------------------------------------------------ #
#                            FEATURE EXTRACTION EXPERIMENT                                         #
# ------------------------------------------------------------------------------------------------ #
class Experiment(BaseExperiment):
    """Performs transfer learning experiments with an optional fine tuning session.

    Args:
        network (Network): A Network object containing the model to be trained.
        config (Config): The experiment configuration object.
        optimizer (type[tf.keras.optimizers.Optimizer]): An optimizer class
        repo (ExperimentRepo): Repository of Weights and Biases experiments
        callbacks (list): List of callbacks
        metrics (list): A list of TensorFlow Keras metrics to track.
        notes (str): Comment about the experiment
        tags (list): Tags used to filter runs.
        force (bool): Whether to force execution if the model already exists in the repository.
    """

    def __init__(
        self,
        network: Network,
        config: Config,
        optimizer: type[tf.keras.optimizers.Optimizer],
        repo: ExperimentRepo,
        callbacks: list = None,
        metrics: list = None,
        notes: str = None,
        tags: list = None,
        force: bool = False,
    ) -> None:
        self._network = network
        self._config = config
        self._optimizer = optimizer
        self._repo = repo
        self._callbacks = callbacks
        self._metrics = metrics
        self._notes = notes
        self._tags = tags
        self._force = force

        self._run_id = None
        self._run = None
        self._filepath = None

        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(logging.INFO)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def filepath(self) -> str:
        return self._filepath

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
        self.announce(network=self._network)

        if (
            self._repo.exists(
                name=self._network.name,
                config_hash=self._config.as_dict()["hash"],
            )
            and not self._force
        ):
            msg = f"Experiment {self._network.name} already exists. Experiment aborted."
            self._logger.warning(msg)
        else:
            # ----------------------------------------------------------------------------------- #
            #                       Instantiate a Weights & Biases run                            #
            # ----------------------------------------------------------------------------------- #
            self._run_id = wandb.util.generate_id()
            self._run = wandb.init(
                id=self._run_id,
                project=self._config.project.name,
                name=self._network.name,
                config=self._config.as_dict(),
                notes=self._notes,
                tags=self._tags,
                resume="allow",
            )
            # ----------------------------------------------------------------------------------- #
            #                                 Add Callbacks                                       #
            # ----------------------------------------------------------------------------------- #
            # Add Weights and Biases callback to track metrics.
            wandb_callback = wandb.keras.WandbMetricsLogger()
            self._callbacks.append(wandb_callback)

            optimizer = self._optimizer(learning_rate=self._config.train.learning_rate)

            # Add a model checkpoint callback if indicated
            if self._config.train.checkpoint:
                # Designate the filepath for the saved model
                self._filepath = self._repo.get_filepath(
                    name=self._network.name,
                    model_id=self._run_id,
                    weights_only=self._config.checkpoint.save_weights_only,
                )

                self._add_checkpoint_callback()
            # ----------------------------------------------------------------------------------- #
            #                              Compile the Model                                      #
            # ----------------------------------------------------------------------------------- #
            self._network.model.compile(
                loss=self._config.train.loss,
                optimizer=optimizer,
                metrics=self._metrics,
            )
            # ----------------------------------------------------------------------------------- #
            #                               Fit the Model                                         #
            # ----------------------------------------------------------------------------------- #
            _ = self._network.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self._config.train.epochs,
                callbacks=self._callbacks,
            )
            # ----------------------------------------------------------------------------------- #
            #                                    FINE TUNING                                      #
            # ----------------------------------------------------------------------------------- #
            if self._config.train.fine_tune:
                # Thaw the model
                self._network.model.trainable = True
                # Reset the learning rate
                tf.keras.backend.set_value(
                    self._network.model.optimizer.lr,
                    self._config.train.learning_rate * 0.1,
                )
                # Resume training
                _ = self._network.model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=self._config.train.epochs,
                    callbacks=self._callbacks,
                )
            # ----------------------------------------------------------------------------------- #
            #                     Register the Model on Weights & Biases                          #
            # ----------------------------------------------------------------------------------- #
            if self._network.register_model and self._config.train.checkpoint:
                self._repo.add(
                    run=self._run,
                    name=self._network.name,
                    filepath=self._filepath,
                )

            # ----------------------------------------------------------------------------------- #
            #                      Create ROC Plot and Confusion Matrix                           #
            # ----------------------------------------------------------------------------------- #
            self.create_plots(model=self._network.model, data=val_ds)

            wandb.finish()

    def _add_checkpoint_callback(self) -> None:
        """Adds a checkpoint callback to the callback instance variable"""

        # Callback will save best model locally.
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self._filepath,
            monitor=self._config.checkpoint.monitor,
            verbose=self._config.checkpoint.verbose,
            save_best_only=self._config.checkpoint.save_best_only,
            save_weights_only=self._config.checkpoint.save_weights_only,
            mode=self._config.checkpoint.mode,
        )
        self._callbacks.append(model_checkpoint_callback)
