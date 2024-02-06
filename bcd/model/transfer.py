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
# Modified   : Monday January 22nd 2024 04:09:47 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import sys
from typing import Union

import tensorflow as tf
import wandb

from bcd.model.callback import Historian
from bcd.model.repo import ModelRepo
from bcd.model.schedule import LearningRateSchedule, ThawSchedule

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
    """Performs feature extraction and fine tuning of a pre-trained model.

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
        thaw_schedule: ThawSchedule,
        learning_rate_schedule: LearningRateSchedule,
        loss: str = "binary_crossentropy",
        monitor: str = "val_loss",
        metrics: list = None,
        fine_tune_epochs: int = 50,
        sessions: int = 10,
        callbacks: Union[list, tf.keras.callbacks.Callback] = None,
    ) -> None:
        self._name = name
        self._train_ds = train_ds
        self._validation_ds = validation_ds
        self._repo = repo
        self._thaw_schedule = thaw_schedule
        self._learning_rate_schedule = learning_rate_schedule
        self._loss = loss
        self._monitor = monitor
        self._metrics = metrics
        self._fine_tune_epochs = fine_tune_epochs
        self._sessions = sessions
        self._callbacks = callbacks

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
                model = self._thaw_schedule(model=model, session=session)

                # Create Optimizer
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self._learning_rate_schedule(session=session)
                )

                # Recompile the model
                model.compile(
                    loss=self._loss,
                    optimizer=optimizer,
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
                _ = model.fit(
                    self._train_ds,
                    validation_data=self._validation_ds,
                    epochs=epochs,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                )

                # Save the model to wandb
                filepath = self._repo.get_filepath(name=self._name, stage=stage)
                artifact = wandb.Artifact(
                    f"{self._name}-{stage}-{wandb.run.id}", type="model"
                )
                artifact.add_file(filepath)
                wandb.log_artifact(artifact, aliases=[stage, "best"])
                wandb.run.link_artifact(
                    artifact, "aistudio/breast_cancer_detection/DenseNet"
                )

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
