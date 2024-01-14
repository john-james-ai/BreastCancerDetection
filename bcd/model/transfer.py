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
# Modified   : Sunday January 14th 2024 09:06:50 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import os
import sys
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf

from bcd.model.visual import X4LearningVisualizer

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=line-too-long


# ------------------------------------------------------------------------------------------------ #
class X4LearnerLRSchedule:
    """Transfer Learning Learning Rate Scheduler

    Learning rate progressively decays with decay values in [2,6]. A list of decay factors, one
    for each session, is created in the linear space between 2 and 6. The returned learning rate
    for a  session is the initial learning rate / 10**learning rate factor for that session.

    Args:
        initial_learning_rate (float): Initial learning rate from feature extraction.
        sessions (int): Number of fine tuning sessions
    """

    def __init__(self, initial_learning_rate: float, sessions: int) -> None:
        self._initial_learning_rate = initial_learning_rate
        self._learning_rate_factors = np.linspace(
            2, 6, endpoint=True, num=sessions
        ).round(0)

    def __call__(self, session) -> float:
        """Returns the learning rate for the session."""
        factor = self._learning_rate_factors[session - 1]
        return self._initial_learning_rate / 10**factor


# ------------------------------------------------------------------------------------------------ #
class X4Learner:
    """Performs transfer learning of a TensorFlow model containing a base pre-trained model.

    Two methods are exposed: extract_features, and fine_tune. The extract_features method trains
    the model on the given data using the designated learning rate. The fine_tune method
    thaws one or more layers in the model, then trains it on a decayed learning rate. Each
    fine tuning session decays the learning rate by a learning_rate_decay factor to mitigate
    catastrophic forgetting.

    Args:
        name (str): Name of the model architecture.
        train_ds (tf.data.Dataset): TensorFlow training dataset.
        val_ds (tf.data.Dataset): TensorFlow validation dataset.
        base_model_layer (int): Index for the base model layer
        learning_rate (float): The learning rate for feature extraction. Default = 0.0001
        patience (int): Number of fine tuning sessions for which no improvement is tolerated. Default = 3.
        metric (str): The metric used to evaluate model fit performance. Default = 'val_accuracy'
        loss (str): The loss function. Default = 'binary_crossentropy'.
        activation (str): Activation function. Default = 'sigmoid'.
        thaw_rate (Union[float, int]): Rate at which layers are thawed. If float, this is the proportion
            of additional base model layers to be frozen each session. If an integer, this is the
            number of layers to be frozen each session.
        callbacks (list): Callback list. Should include an early stopping callback if max_epochs and/or
            feature_extraction_epochs are significant.

    """

    __model_directory = "models/"

    def __init__(
        self,
        name: str,
        model: tf.keras.Model,
        base_model_layer: int,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        learning_rate: float = 0.0001,
        patience: int = 3,
        metric: str = "val_loss",
        loss: str = "binary_crossentropy",
        activation: str = "sigmoid",
        callbacks: list = None,
        visualizer: type[X4LearningVisualizer] = X4LearningVisualizer,
    ) -> None:
        self._name = name
        self._model = model
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._base_model_layer = base_model_layer
        self._learning_rate = learning_rate
        self._patience = patience
        self._metric = metric
        self._loss = loss
        self._activation = activation
        self._callbacks = callbacks
        self._visualizer = visualizer(name=self._name)

        self._n_layers = len(self._model.layers[self._base_model_layer].layers)
        self._best_score = np.inf if "loss" in self._metric else 0
        self._early_stop_counter = 0

        self._initial_epoch = None
        self._model_scores = []
        self._features_extracted = False
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

        self._logger.setLevel(level=logging.DEBUG)

    @property
    def scores(self) -> pd.DataFrame:
        if not self._features_extracted:
            msg = "Model has not been created"
            self._logger.exception(msg)
        else:
            return pd.DataFrame(data=self._model_scores)

    # ------------------------------------------------------------------------------------------------ #
    def extract_features(self, epochs: int = 100) -> None:
        """Performs the feature extraction phase of transfer learning

        Args:
            epochs (int): Number of epochs to execute
        """

        modelname = self._name + "_feature_extraction"
        msg = f"Starting feature extraction stage for {modelname}."
        self._logger.info(msg)

        history = self._model.fit(
            self._train_ds,
            epochs=epochs,
            validation_data=self._val_ds,
            callbacks=[self._callbacks],
        )

        self._visualizer(history=history)

        # Save the last feature extraction epoch for fine tune reset
        self._initial_epoch = history.epoch[-1]

        self.save_model(modelname=modelname, model=self._model)

        score = self._get_best_score(history)
        self._record_best_score(modelname=modelname, score=score)

        self._features_extracted = True

        msg = f"Completed feature extraction stage for {modelname}."
        self._logger.info(msg)

        return self._model

    # ------------------------------------------------------------------------------------------------ #
    def fine_tune(
        self,
        strategy: str = "resume",
        epochs: int = 10,
        sessions: int = 10,
        thaw_rate: Union[float, int] = 0.05,
        lrschedule: type[X4LearnerLRSchedule] = X4LearnerLRSchedule,
    ) -> None:
        """Performs iterative fine tuning using gradual unfreezing of the base model.

        Choices of fine tuning strategy include: 'restart', and 'resume'. The 'restart' strategy
        starts each session with the feature extraction model. The 'resume' strategy starts
        with the prior model and epochs.

        Args:
            strategy (str): Either 'restart' or 'resume'. Default is 'resume'.
            epochs (int): Number of epochs per session. Default = 10
            sessions (int): Number of fine tuning sessions to execute. Default is 10
            learning_rate_decay (int): Factor by which the learning rate is reduced each session.
            thaw_rate (Union[float, int]): Rate by which layers are thawed. This can be a raw
                integer or a float proportion of base model layers. Default = 0.05.
        """

        # Initialization
        self._validate()
        session = 0
        history = None
        lrs = lrschedule(initial_learning_rate=self._learning_rate, sessions=sessions)

        modelname = self._name + "_feature_extraction"
        self._model = self.load_model(modelname=modelname)

        while session < sessions:
            session += 1

            learning_rate = lrs(session=session)

            modelname = self._name + "_fine_tune_" + strategy + "_" + str(session)

            msg = f"Fine tuning {modelname} session #{session} with {strategy} strategy with learning rate {learning_rate:e}."
            self._logger.info(msg)

            if "restart" in strategy:
                initial_epoch = self._initial_epoch
            else:
                try:
                    initial_epoch = history.epoch[-1]
                except AttributeError:
                    initial_epoch = self._initial_epoch

            self._model = self._thaw(
                modelname=modelname,
                model=self._model,
                session=session,
                thaw_rate=thaw_rate,
            )

            self._model.compile(
                loss=self._loss,
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=[self._metric],
            )
            total_epochs = epochs + initial_epoch
            history = self._model.fit(
                self._train_ds,
                epochs=total_epochs,
                validation_data=self._val_ds,
                initial_epoch=initial_epoch,
                callbacks=[self._callbacks],
            )

            self._visualizer(history=history)

            self.save_model(modelname=modelname, model=self._model)

            score = self._get_best_score(history)
            self._record_best_score(modelname=modelname, score=score)

            if not self._is_improving(modelname=modelname, score=score):
                break

    # ------------------------------------------------------------------------------------------------ #
    #                                           MODEL IO                                               #
    # ------------------------------------------------------------------------------------------------ #
    def save_model(self, modelname: str, model: tf.keras.Model) -> None:
        """Saves a TensorFlow model to file.

        Args:
            modelname (str): Name of the model.
            model (tf.keras.Model): Model to save.
        """
        filename = modelname + ".keras"
        filepath = os.path.join(self.__model_directory, self._name, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model.save(filepath)
        msg = f"Saved {modelname} to {filepath}."
        self._logger.debug(msg)

    def load_model(self, modelname: str) -> tf.keras.Model:
        """Loads a TensorFlow model from file.

        Args:
            modelname (str): Name of the model.

        Returns: tensorflow.keras.Model
        """
        filename = modelname + ".keras"
        filepath = os.path.join(self.__model_directory, self._name, filename)
        model = tf.keras.models.load_model(filepath, compile=False)
        msg = f"Loaded {modelname} from {filepath}"
        self._logger.debug(msg)
        return model

    # ------------------------------------------------------------------------------------------------ #
    def _thaw(
        self,
        modelname: str,
        model: tf.keras.Model,
        session: int,
        thaw_rate: Union[float, int],
    ) -> tf.keras.Model:
        """Thaws n top layers of a TensorFlow model"""

        if thaw_rate > 1:
            n = session * thaw_rate
        else:
            n = session * int(thaw_rate * self._n_layers)

        model.layers[self._base_model_layer].trainable = True
        for layer in model.layers[self._base_model_layer].layers[:-n]:
            layer.trainable = False

        msg = f"Model {modelname} has {n} thawed layers and {self._n_layers-n} frozen layers."
        self._logger.info(msg)

        return model

    # ------------------------------------------------------------------------------------------------ #
    def _get_best_score(self, history: tf.keras.callbacks.History) -> float:
        if "loss" in self._metric:
            return min(history.history[self._metric])
        else:
            return max(history.history[self._metric])

    # ------------------------------------------------------------------------------------------------ #
    def _record_best_score(self, modelname: str, score: float) -> float:
        """Records best model score."""
        d = {modelname: score}
        self._model_scores.append(d)

        msg = f"Best {self._metric} score for {modelname} is {round(score,4)}"
        self._logger.info(msg)

    # ------------------------------------------------------------------------------------------------ #
    def _validate(self) -> None:
        """Ensures the object is in a valid state prior to fine tuning"""

        if not self._features_extracted:
            msg = "Feature extraction must occur before fine tuning."
            self._logger.exception(msg)
            raise RuntimeError(msg)

    # ------------------------------------------------------------------------------------------------ #
    def _is_improving(self, modelname: str, score: float) -> bool:
        if "loss" in self._metric:
            if score < self._best_score:
                msg = f"Fine tuning {modelname} {self._metric} improved to {round(score,4)} from {round(self._best_score,4)}."
                self._best_score = score
                self._early_stop_counter = 0
            else:
                msg = f"Fine tuning {modelname} {self._metric} did NOT improve {round(score)}. Performance dropped by {round((score-self._best_score)/self._best_score*100,4)}%"
                self._early_stop_counter += 1
        else:
            if score >= self._best_score:
                msg = f"Fine tuning {modelname} {self._metric} improved to {round(score,4)} from {round(self._best_score,4)}."
                self._best_score = score
                self._early_stop_counter = 0
            else:
                msg = f"Fine tuning {modelname} {self._metric} did NOT improve {round(score)}. Performance dropped by {round((self._best_score-score)/self._best_score*100,4)}%"
                self._early_stop_counter += 1
        logging.info(msg)
        return self._early_stop_counter < self._patience
