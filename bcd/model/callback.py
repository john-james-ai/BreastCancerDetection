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
# Modified   : Friday January 19th 2024 07:51:27 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""TensorFlow Callback Module """
import os
import pickle
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")

warnings.simplefilter(action="ignore", category=FutureWarning)


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=unused-argument, consider-iterating-dictionary
# ------------------------------------------------------------------------------------------------ #
class Historian(tf.keras.callbacks.Callback):
    """Encapsulates training history."""

    __directory = "models"

    def __init__(self, name: str) -> None:
        self._name = name
        self._history = {}
        self._last_epoch = None
        self._start = None
        self._stop = None
        self._session = None
        filename = f"{name}_history.pkl"
        self._filepath = os.path.join(self.__directory, self._name, filename)

    @property
    def last_epoch(self) -> int:
        return self._last_epoch

    @property
    def last_session(self) -> int:
        return self._session

    @property
    def summary(self) -> pd.DataFrame:
        try:
            return self._summarize()
        except KeyError:
            return None

    def on_session_begin(self, session: int) -> None:
        self._session = session
        self._history[session] = {}

    def on_epoch_begin(self, epoch, logs: dict = None) -> None:
        self._start = datetime.now()

    def on_epoch_end(self, epoch, logs=None) -> None:
        self._last_epoch = epoch
        self._stop = datetime.now()
        logs["duration"] = (self._stop - self._start).total_seconds()
        for k, v in logs.items():
            self._history[self._session].setdefault(k, []).append(v)

    def on_train_end(self, logs: dict = None) -> None:
        self.save()

    def save(self) -> None:
        with open(self._filepath, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath) -> None:
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def plot_learning_curves(self) -> None:
        """Plots learning curve for all epochs in the history.

        Plots learning curve. If mode is 'a', the history is appended
        to the existing history and the plot will cover all epochs
        in the cumulative history. If mode is 'w', only the scores
        from the history parameter are plotted.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        summary = self.summary

        df_train_loss = summary[["epoch", "loss"]].copy()
        df_train_loss.loc[:, "dataset"] = "train"
        df_val_loss = summary[["epoch", "val_loss"]].copy()
        df_val_loss.loc[:, "dataset"] = "validation"
        df_val_loss.rename(columns={"val_loss": "loss"}, inplace=True)
        df_loss = pd.concat([df_train_loss, df_val_loss], axis=0)

        df_train_accuracy = summary[["epoch", "accuracy"]].copy()
        df_train_accuracy.loc[:, "dataset"] = "train"
        df_val_accuracy = summary[["epoch", "val_accuracy"]].copy()
        df_val_accuracy.loc[:, "dataset"] = "validation"
        df_val_accuracy.rename(columns={"val_accuracy": "accuracy"}, inplace=True)
        df_accuracy = pd.concat([df_train_accuracy, df_val_accuracy], axis=0)

        sns.lineplot(data=df_loss, x="epoch", y="loss", hue="dataset", ax=axes[0])
        title = f"{self._name.capitalize()}\nTraining and Validation Loss"
        axes[0].set_title(title)

        sns.lineplot(
            data=df_accuracy, x="epoch", y="accuracy", hue="dataset", ax=axes[1]
        )
        title = f"{self._name.capitalize()}\nTraining and Validation Accuracy"
        axes[1].set_title(title)

        # Add session lines
        s = summary[["session"]]
        idx = np.unique(s, return_index=True)[1] + 1
        for session in idx:
            axes[0].axvline(x=session, color="r")
            axes[1].axvline(x=session, color="r")

        fig.suptitle(
            f"{self._name.capitalize()} Learning Curves\nTransfer Learning\nFeature Extraction and Fine Tuning Sessions (Red)"
        )
        plt.tight_layout()
        plt.show()

    def _summarize(self) -> pd.DataFrame:
        data = {}
        for session, metrics in self._history.items():
            for metric, values in metrics.items():
                for epoch, value in enumerate(values):
                    data.setdefault("session", []).append(session)
                    data.setdefault("epoch", []).append(epoch)
                    data.setdefault("metric", []).append(metric)
                    data.setdefault("value", []).append(value)
        df = pd.DataFrame(data)
        df = df.pivot(index=["session", "epoch"], columns="metric", values="value")
        df = df.rename_axis(mapper=None, axis=1)
        df = df.reset_index()
        df["epoch"] = range(1, len(df) + 1)
        try:
            df = df.drop(columns=["epochs"])
        except KeyError:
            pass
        return df


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

        save_freq (str): "epoch" or integer. When using "epoch", the callback saves the
        model after each epoch. When using integer, the callback saves the model at end
        of this many batches. If the Model is compiled with steps_per_execution=N,
        then the saving criteria will be checked every Nth batch. Note that if the
        saving isn't aligned to epochs, the monitored metric may potentially be
        less reliable (it could reflect as little as 1 batch, since the metrics
        get reset every epoch). Defaults to "epoch".

        verbose (int): Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
        displays messages when the callback takes an action.
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
