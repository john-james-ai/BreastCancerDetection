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
# Modified   : Monday February 12th 2024 12:15:08 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
# pylint: disable=wrong-import-order
# ------------------------------------------------------------------------------------------------ #
"Experiment Module"
import logging
import os

import numpy as np
import tensorflow as tf
import wandb
from dotenv import load_dotenv

from bcd.model.network.base import Network
from bcd.model.repo import ModelRepo
from bcd.model.schedule import FineTuneSchedule
from bcd.utils.hash import dict_hash

# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member


# ------------------------------------------------------------------------------------------------ #
#                                            EXPERIMENT                                            #
# ------------------------------------------------------------------------------------------------ #
class Experiment:
    """Performs transfer learning experiments.

    Transfer learning experiments are performed on models based upon an underlying pretrained
    model in which the top layers are replaced with a new classification head. All layers except
    those of the classification head are frozen and the model is trained to convergence up
    to a maximum number of epochs designated in the config parameter.

    Args:
        network (list): A Network object containing a TensorFlow models to be trained.
        config (dict): The experiment configuration.

        optimizer (tf.keras.optimizers): A TensorFlow Keras optimizer instance.
        repo (ModelRepo): Repository containing all models.
        fine_tune_schedule (FineTuneSchedule): Schedule for fine tuning. Optional.
        callbacks (list): A list of TensorFlow keras callbacks
        metrics (list): A list of TensorFlow Keras metrics to track.
        force (bool): Whether to force execution if the model already exists in the repository.
    """

    def __init__(
        self,
        network,
        config: dict,
        optimizer: type[tf.keras.optimizers.Optimizer],
        repo: ModelRepo,
        fine_tune_schedule: FineTuneSchedule = None,
        callbacks: list = None,
        metrics: list = None,
        force: bool = False,
    ) -> None:
        self._network = network
        self._config = config
        self._optimizer = optimizer
        self._repo = repo
        self._fine_tune_schedule = fine_tune_schedule
        self._callbacks = callbacks
        self._metrics = metrics
        self._force = force

        self._train_ds = None
        self._val_ds = None

        self._entity = os.getenv("WANDB_ENTITY")

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

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

        self._train_ds = train_ds
        self._val_ds = val_ds

        experiment_config = self._update_config(self._network)
        if (
            self._repo.experiment_exists(
                name=self._network.name, config=experiment_config
            )
            and not self._force
        ):
            msg = f"Experiment {self._network.name} already exists. Experiment skipped."
            self._logger.info(msg)
        else:
            # Generate a run_id for the next run.
            run_id = wandb.util.generate_id()
            # Instantiate a wandb run and callback
            run = wandb.init(
                id=run_id,
                project=experiment_config["project"],
                name=self._network.name,
                config=experiment_config,
                tags=[self._network.architecture, self._network.base_model.name],
                resume="allow",
            )

            callbacks = self._configure_callbacks(
                run=run, network=self._network, config=experiment_config
            )

            self._network = self._extract_features(
                run=run,
                network=self._network,
                config=experiment_config,
                callbacks=callbacks,
            )
            if self._fine_tune_schedule is not None:
                self._fine_tune(
                    run=run,
                    network=self._network,
                    config=experiment_config,
                    callbacks=callbacks,
                )

            val_accuracy = run.summary["epoch/val_accuracy"]

            wandb.finish()

            return val_accuracy

    def _extract_features(
        self, run: wandb.run, network: Network, config: dict, callbacks: list
    ) -> str:
        """Conduct the feature extraction stage.

        Args:
            run (wandb.run): Weights & Biases run object.
            network (Network): Network to be trained.
            config (dict): Experiment configuration.
            callbacks (list): List of callbacks.

        Returns a string containing the Weights & Biases run id.
        """

        # For memory and performance efficiency, release the
        # global state keras maintains to implement the
        # Functional API.
        tf.keras.backend.clear_session()

        # Compile the model
        network.model.compile(
            loss=config["training"]["loss"],
            optimizer=self._optimizer(
                learning_rate=config["training"]["learning_rate"]
            ),
            metrics=self._metrics,
        )

        # Fit the model
        _ = network.model.fit(
            self._train_ds,
            validation_data=self._val_ds,
            epochs=config["training"]["epochs"],
            callbacks=callbacks,
        )

        # Register the model as an artifact on wandb if specified.
        if network.register_model:
            self._repo.add(run=run, name=network.name, model=network.model)

        # Create and load plots to Weights and Biases
        self._create_plots(model=network.model, data=self._val_ds)

        return network

    def _fine_tune(
        self, run: wandb.run, network: Network, config: dict, callbacks: list
    ) -> None:
        """Performs fine tuning of the network according to the fine tune schedule"""
        # Create the fine tune schedule for the network
        self._fine_tune_schedule.create(network=network)
        # Iterate over the sessions
        for session in self._fine_tune_schedule.sessions:
            # Thaw layers of the network, update the optimizer's learning rate and get epochs to train.
            network = self._fine_tune_schedule.thaw(network=network, session=session)
            optimizer = self._fine_tune_schedule.update_learning_rate(
                session=session, optimizer=self._optimizer
            )
            epochs = self._fine_tune_schedule.get_epochs(session=session)

            # Clear session before compiling
            tf.keras.backend.clear_session()

            # Compile the model with the thawed layers.
            network.model.compile(
                loss=config["training"]["loss"],
                optimizer=optimizer,
                metrics=self._metrics,
            )

            # Fit the model
            _ = network.model.fit(
                self._train_ds,
                validation_data=self._val_ds,
                epochs=epochs,
                callbacks=callbacks,
            )

            # Register the model as an artifact on wandb if specified.
            if network.register_model:
                self._repo.add(run=run, name=network.name, model=network.model)

            # Create and load plots to Weights and Biases
            self._create_plots(model=network.model, data=self._val_ds)

    def _create_plots(self, model: tf.keras.Model, data: tf.data.Dataset) -> None:
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

    def _update_config(self, network: Network) -> dict:
        """Adds network and (optionally) fine tune config and creates a hash. ."""
        config = self._config
        config["network"] = network.config
        if self._fine_tune_schedule is not None:
            config["fine_tune"] = self._fine_tune_schedule.as_dict()
        config["hash"] = dict_hash(dictionary=config)
        return config

    def _configure_callbacks(
        self, run: wandb.run, network: Network, config: dict
    ) -> list:
        """Configures the callbacks for the run."""
        # Extract callbacks
        callbacks = self._callbacks

        # Create a wandb callback to track metrics
        wandb_callback = wandb.keras.WandbMetricsLogger()
        callbacks.append(wandb_callback)

        # create the model checkpoint callback for the network.
        filepath = self._repo.get_filepath(name=network.name, run_id=run.id)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=config["checkpoint"]["monitor"],
            verbose=config["checkpoint"]["verbose"],
            save_best_only=config["checkpoint"]["save_best_only"],
            save_weights_only=config["checkpoint"]["save_weights_only"],
            mode=config["checkpoint"]["mode"],
        )
        callbacks.append(model_checkpoint_callback)
        return callbacks
