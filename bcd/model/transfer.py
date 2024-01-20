#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/transfer.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday January 13th 2024 08:37:37 pm                                              #
# Modified   : Friday January 19th 2024 04:38:42 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import sys
from typing import Union

import numpy as np
import tensorflow as tf

from bcd.model.callback import Historian
from bcd.model.repo import ModelRepo

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=line-too-long


# ------------------------------------------------------------------------------------------------ #
def thaw(
    model: tf.keras.Model,
    base_model_layer: int,
    n: int,
) -> tf.keras.Model:
    """Thaws n top layers of a TensorFlow model

    Args:
        model (tf.keras.Model): Model to thaw
        base_model_layer (int): The layer containing the base model
        n (int): Top number of layers in base model to thaw.

    """
    model.layers[base_model_layer].trainable = True
    for layer in model.layers[base_model_layer].layers[:-n]:
        layer.trainable = False

    return model


# ================================================================================================ #
#                                       FINE TUNER                                                 #
# ================================================================================================ #
class FineTuner:
    """Performs fine tuning of a model that has converged under feature extraction.

    Fine tuning occurs up to a maximum number of sessions, subject to early stop. If the
    monitored score hasn't improved in patience sessions, fine tuning stops. Fine tuning is
    performed according to a thaw schedule. The thaw schedule contains the number of layers
    to thaw and the associated learning rate for each session.

    Args:
        name (str): Name of the model
        train_ds (tf.data.Dataset)
    """

    def __init__(
        self,
        name: str,
        train_ds: tf.data.Dataset,
        validation_ds: tf.data.Dataset,
        repo: ModelRepo,
        loss: str = "binary_crossentropy",
        monitor: str = "val_loss",
        metrics: list = None,
        patience: int = 3,
        min_delta: float = 0.0001,
        initial_learning_rate: float = 1e-5,
        final_learning_rate: float = 1e-10,
        fine_tune_epochs: int = 50,
        sessions: int = 10,
        callbacks: Union[list, tf.keras.callbacks.Callback] = None,
    ) -> None:
        self._name = name
        self._train_ds = train_ds
        self._validation_ds = validation_ds
        self._repo = repo
        self._loss = loss
        self._monitor = monitor
        self._metrics = metrics
        self._patience = patience
        self._min_delta = min_delta
        self._initial_learning_rate = initial_learning_rate
        self._final_learning_rate = final_learning_rate
        self._fine_tune_epochs = fine_tune_epochs
        self._sessions = sessions
        self._callbacks = callbacks

        self._best_score = np.inf if "loss" in self._monitor else 0

        self._base_model_layer = None
        self._thaw_layers = []
        self._learning_rates = []
        self._early_stop_counter = 0

        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(level=logging.DEBUG)

    # -------------------------------------------------------------------------------------------- #
    def tune(
        self,
        model: tf.keras.Model,
        historian: Historian,
        base_model_layer: int,
        force: bool = False,
    ) -> None:
        """Performs fine tuning of the model

        This method takes a model that has been trained to conversion during
        feature extraction. A gradual fine tuning schedule is created which lists the
        layers to unfreeze in session groups from 1 to all layers on a log scale.
        Each session, the model is trained for sessions epochs, subject to early
        stopping. At the end of each session, the monitored score is evaluated.
        If the score hasn't improved in patience sessions,  fine tuning
        stops.

        Args:
            model (tf.keras.Model): The model to be fine tuned.
            history (History): Historian object.
            base_model_layer (int): Layer of the base model.
            last_epoch (int): Last epoch
            force (bool): Whether to force training if the model already exists.
        """
        session = 0
        # Create a layer thaw schedule based upon the number of sessions and
        # the number of layers in the base model. The number of layers to
        # thaw in each session grows logarithmically from 1 to num_layers in model
        # for 'session' values.
        self._create_thaw_schedule(model=model, base_model_layer=base_model_layer)
        # Add the historian to the callbacks.
        self._callbacks = self._add_callback(self._callbacks, historian)

        while session < self._sessions:
            session += 1
            stage = f"fine_tuning_session_{session}"
            # Get the model from the repository if it already exists
            # and we're not forcing.
            if self._repo.exists(name=self._name, stage=stage) and not force:
                model = self._repo.get(name=self._name, stage=stage)
            else:
                # Start session
                historian.on_session_begin(session=session)
                # Remove existing checkpoints if they exist
                self._repo.remove(name=self._name, stage=stage)

                # Thaw top n layers according to thaw schedule
                model = self._thaw(
                    model=model,
                    base_model_layer=base_model_layer,
                    n=self._thaw_layers[session - 1],
                )

                print("\n")
                msg = f"Thawing {self._thaw_layers[session-1]} layers and training with {self._learning_rates[session-1]} learning rate."
                self._logger.info(msg)

                # Recompile the model
                model.compile(
                    loss=self._loss,
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self._learning_rates[session - 1]
                    ),
                    metrics=self._metrics,
                )

                # Summarize the model
                model.summary()

                # Extract last epoch from historian
                initial_epoch = historian.last_epoch + 1
                epochs = initial_epoch + self._fine_tune_epochs

                # Create checkpoint callback
                checkpoint_callback = self._repo.create_callback(
                    name=self._name, stage=stage
                )

                # Add the checkpoint callback to the callback list.
                callbacks = self._add_callback(self._callbacks, checkpoint_callback)

                # Fine tune the model
                history = model.fit(
                    self._train_ds,
                    validation_data=self._validation_ds,
                    epochs=epochs,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                )

                # Obtain best score
                best_score = self._get_best_score(history=history)

                if not self._has_improved(score=best_score):
                    break

    # -------------------------------------------------------------------------------------------- #
    def _add_callback(
        self,
        callbacks: Union[None, tf.keras.callbacks.Callback, list],
        callback: tf.keras.callbacks.Callback,
    ) -> list:
        # Create checkpoint for the session and add it to sessions
        orig_callbacks = callbacks.copy()

        if orig_callbacks is None:
            return [callback]
        elif isinstance(orig_callbacks, list):
            orig_callbacks.append(callback)
            return orig_callbacks
        else:
            return [orig_callbacks, callback]

    # -------------------------------------------------------------------------------------------- #
    def _create_thaw_schedule(
        self, model: tf.keras.Model, base_model_layer: int
    ) -> None:
        """Computes a schedule of layers to thaw."""
        n_layers = len(model.layers[base_model_layer].layers)
        self._thaw_layers = list(
            np.geomspace(
                start=1, stop=n_layers, endpoint=True, num=self._sessions
            ).astype(int)
        )
        self._learning_rates = list(
            np.geomspace(
                start=self._initial_learning_rate,
                stop=self._final_learning_rate,
                endpoint=True,
                num=self._sessions,
            )
        )

    # ------------------------------------------------------------------------------------------------ #
    def _thaw(
        self,
        model: tf.keras.Model,
        base_model_layer: int,
        n: int,
    ) -> tf.keras.Model:
        """Thaws n top layers of a TensorFlow model

        Args:
            model (tf.keras.Model): Model to thaw
            base_model_layer (int): The layer containing the base model
            n (int): Top number of layers in base model to thaw.

        """

        model.layers[base_model_layer].trainable = True
        for layer in model.layers[base_model_layer].layers[:-n]:
            layer.trainable = False

        return model

    # -------------------------------------------------------------------------------------------- #
    def _get_best_score(self, history: tf.keras.callbacks.History) -> float:
        if "loss" in self._monitor:
            return min(history.history[self._monitor])
        else:
            return max(history.history[self._monitor])

    # -------------------------------------------------------------------------------------------- #
    def _has_improved(self, score: float) -> bool:
        if "loss" in self._monitor:
            if score < self._best_score - self._min_delta:
                msg = f"Fine tuning {self._name} {self._monitor} improved to {round(score,4)} from {round(self._best_score,4)}."
                self._best_score = score
                self._early_stop_counter = 0
            else:
                msg = f"Fine tuning {self._name} {self._monitor} did NOT improve {round(score)}. Performance dropped by {round((score-self._best_score)/self._best_score*100,4)}%"
                self._early_stop_counter += 1
        else:
            if score > self._best_score + self._min_delta:
                msg = f"Fine tuning {self._name} {self._monitor} improved to {round(score,4)} from {round(self._best_score,4)}."
                self._best_score = score
                self._early_stop_counter = 0
            else:
                msg = f"Fine tuning {self._name} {self._monitor} did NOT improve {round(score)}. Performance dropped by {round((self._best_score-score)/self._best_score*100,4)}%"
                self._early_stop_counter += 1
        self._logger.info(msg)
        return self._early_stop_counter < self._patience
