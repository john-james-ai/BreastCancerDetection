#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/callback.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday November 5th 2023 11:35:52 am                                                #
# Modified   : Monday January 15th 2024 07:20:50 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""TensorFlow Callback Module """
import os
from datetime import datetime

import tensorflow as tf
from keras.callbacks import Callback

# ------------------------------------------------------------------------------------------------ #


class DurationCallback(Callback):
    """Records duration of each epoch in the model history."""

    def __init__(self, name: str = "duration"):
        super().__init__()
        self._start = None
        self._end = None
        self._name = name
        self._total_duration = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._start = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        self._end = datetime.now()
        duration = (self._end - self._start).total_seconds()
        self._total_duration += duration
        logs[self._name] = duration
        logs["total_duration"] = self._total_duration


# ------------------------------------------------------------------------------------------------ #
class CheckpointCallbackFactory:
    """ModelCheckpoint Callback Factory

    Args:
        location (str): Base location for all model checkpoints.

        monitor: The metric name to monitor. Typically the metrics are set by the Model.compile
        method. Note: Prefix the name with "val_" to monitor validation metrics. Use "loss" or
        "val_loss" to monitor the model's total loss. If you specify metrics as strings, like
        "accuracy", pass the same string (with or without the "val_" prefix). If you pass
        metrics.Metric objects, monitor should be set to metric.name If you're not sure about the
        metric names you can check the contents of the history.history dictionary returned by
        history = model.fit() Multi-output models set additional prefixes on the metric names.

        mode (str): one of {"auto", "min", "max"}. If save_best_only=True, the decision to overwrite
        the current save file is made based on either the maximization or the minimization of the
        monitored quantity. For val_acc, this should be "max", for val_loss this should be "min",
        etc. In "auto" mode, the mode is set to "max" if the quantities monitored are "acc" or start
        with "fmeasure" and are set to "min" for the rest of the quantities.

        save_weights_only (bool): if True, then only the model's weights will be saved
        (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).

        save_freq (str): "epoch" or integer. When using "epoch", the callback saves the model after each
        epoch. When using integer, the callback saves the model at end of this many batches. If the
        Model is compiled with steps_per_execution=N, then the saving criteria will be checked every
        Nth batch. Note that if the saving isn't aligned to epochs, the monitored metric may
        potentially be less reliable (it could reflect as little as 1 batch, since the metrics get
        reset every epoch). Defaults to "epoch".

        verbose (int): Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when
        the callback takes an action.
    """

    def __init__(
        self,
        location: str,
        monitor: str = "val_loss",
        mode: str = "auto",
        save_weights_only: bool = False,
        save_best_only: bool = True,
        save_freq: str = "epoch",
        verbose: int = 1,
    ) -> None:
        self._location = location
        self._monitor = monitor
        self._mode = mode
        self._save_weights_only = save_weights_only
        self._save_best_only = save_best_only
        self._save_freq = save_freq
        self._verbose = verbose

    def __call__(self, name: str, stage: str) -> tf.keras.callbacks.ModelCheckpoint:
        """Creates and returns the ModelCheckpoint callback.

        Args:
            name (str): The name of the model
            stage (str): Brief description of model stage.
        """
        filename = f"{name}_{stage}_"
        filename = filename + "{epoch:02d}-val_loss_{val_loss:.2f}.keras"
        filepath = os.path.join(self._location, name, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=self._monitor,
            mode=self._mode,
            save_weights_only=self._save_weights_only,
            save_best_only=self._save_best_only,
            save_freq=self._save_freq,
            verbose=self._verbose,
        )
