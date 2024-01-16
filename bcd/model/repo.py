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
# Modified   : Tuesday January 16th 2024 01:43:00 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import os
import re
from datetime import datetime
from glob import glob
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf

# ------------------------------------------------------------------------------------------------ #
MODEL_DIR = "models/"
REGISTRY_FILENAME = "registry.csv"


# ------------------------------------------------------------------------------------------------ #
class ModelRepo:
    """Repository for models.

    Args:
        location (str): Base directory for models.
    """

    def __init__(self, location: str = None) -> None:
        self._location = location or MODEL_DIR
        self._registry_filepath = os.path.join(self._location, REGISTRY_FILENAME)
        try:
            self._registry = pd.read_csv(self._registry_filepath)
        except FileNotFoundError:
            self._registry = None

        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(level=logging.DEBUG)

    @property
    def registry(self) -> pd.DataFrame:
        if self._registry is None:
            try:
                self._registry = pd.read_csv(self._registry_filepath)
            except FileNotFoundError:
                self._registry = None
        return self._registry

    def add(
        self,
        model: tf.keras.Model,
        name: str,
        stage: str,
        history: tf.keras.callbacks.History,
        monitor: str = "val_loss",
    ) -> None:
        """Adds a model to the repository

        Args:
            model (tf.keras.Model): Model to be saved to the repository.
            name (str): Name of the model
            stage (str): Stage of the model. Examples would be 'feature_extraction', or 'fine_tuning_session_2'.
            history (tf.keras.callbacks.History): A history object returned from the fit method
            monitor (str): The metric that was monitored
        """
        dt = datetime.now()
        version = self._get_version(name=name)
        score, epoch = self._get_best_score(history=history, monitor=monitor)
        epochs = self._get_epochs(history=history)
        filepath = self._format_filepath(
            name=name,
            stage=stage,
            version=version,
            monitor=monitor,
            score=score,
            epoch=epoch,
            epochs=epochs,
            dt=dt,
        )

        self.save(model=model, filepath=filepath)

        data = {
            "name": name,
            "version": version,
            "stage": stage,
            "monitor": monitor,
            "score": score,
            "epoch": epoch,
            "epochs": epochs,
            "filepath": filepath,
            "datetime": dt,
        }
        self._update_registry(data=data)

    def get(self, name: str, version: int) -> tf.keras.Model:
        """Returns a model given the name and version

        Args:
            name (str): Name of the model
            version (int): Version of the model.
        """

        if self.registry is not None:
            registry = self.registry
            filepath = registry.loc[
                (registry["name"] == name) & (registry["version"] == version),
                "filepath",
            ].values[0]
            if filepath is not None:
                return self.load(filepath=filepath)

        msg = f"Model {name} version {version} not found."
        self._logger.exception(msg)
        raise FileNotFoundError(msg)

    def remove(self, name: str, version: int = None) -> None:
        """Removes a model or model version from the repository

        Args:
            name (str): Model name
            version (int): Model version.
        """
        if version is None:
            self._remove_model(name=name)
        else:
            self._remove_model_version(name=name, version=version)

    def save(self, model: tf.keras.Model, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath))
        model.save(filepath)

    def load(self, filepath: str) -> tf.keras.Model:
        try:
            return tf.keras.models.load_model(filepath)
        except FileNotFoundError as exc:
            msg = f"Model not found at {filepath}"
            self._logger.exception(msg)
            raise FileNotFoundError(msg) from exc

    def _format_filepath(
        self,
        name: str,
        version: int,
        monitor: str,
        score: float,
        epoch: int,
        epochs: int,
        dt: datetime,
    ) -> str:
        dtf = dt.strftime("%Y%m%d_T%H%M%S")
        filename = f"{name}_v{version}_{monitor}-{round(score,4)}_epoch_{epoch}_of_{epochs}_epochs_{dtf}.keras"
        return os.path.join(self._location, name, filename)

    def _get_version(self, name: str) -> int:
        """Returns the next available version number for a model name."""
        df = self.registry
        try:
            return df["name"].loc[df["name"] == name].count() + 1
        except TypeError:
            return 1

    def _get_best_score(
        self, history: tf.keras.callbacks.History, monitor: str = "val_loss"
    ) -> Union[float, int]:
        """Returns the best score and best epoch from history"""
        if "loss" in monitor:
            score = min(history.history[monitor])
            epoch = np.argmin(history.history[monitor]) + 1
        else:
            score = max(history.history[monitor])
            epoch = np.argmax(history.history[monitor]) + 1
        return score, epoch

    def _get_epochs(self, history: tf.keras.callbacks.History) -> int:
        """Returns the number of epochs in the history."""
        return len(history.history["val_loss"])

    def _update_registry(self, data: dict) -> None:
        """Updates the registry with the additional data."""
        df = pd.DataFrame(data=data, index=[0])
        if self.registry is None:
            self._registry = df
        else:
            self._registry = pd.concat([self._registry, df], axis=0)
        self._registry.to_csv(self._registry_filepath, index=False)

    def _remove_model(self, name: str) -> None:
        """Removes a model and its versions from the repository"""
        if self.registry is not None:
            registry = self.registry
            rows = registry.loc[registry["name"] == name]
            if len(rows) > 0:
                for filepath in rows["filepath"]:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        msg = f"Model {name} at {filepath} has been removed from the model repository."
                        self._logger.info(msg)
                    else:
                        msg = f"Model {name} was not found."
                        self._logger.warning(msg)
                self._registry = registry.loc[registry["name"] != name]
                self._registry.to_csv(self._registry_filepath, index=False)
            else:
                msg = f"No models named {name} were found."
                self._logger.warning(msg)
        else:
            msg = f"No models named {name} were found."
            self._logger.warning(msg)

    def _remove_model_version(self, name: str, version: int) -> None:
        """Removes a specific model version from the repository"""
        if self._registry is not None:
            registry = self.registry
            filepath = registry.loc[
                (registry["name"] == name) & (registry["version"] == version),
                "filepath",
            ].values[0]
            if filepath is not None:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    msg = f"Model {name} v{version} at {filepath} has been removed from the model repository."
                    self._logger.info(msg)
                else:
                    msg = f"Model {name}-v{version} was not found."
                    self._logger.warning(msg)
                self._registry = registry.loc[
                    (registry["name"] != name) & (registry["version"] != version)
                ]
                self._registry.to_csv(self._registry_filepath, index=False)
            else:
                msg = f"Model {name}-v{version} was not found."
                self._logger.warning(msg)
        else:
            msg = f"Model {name}-v{version} was not found."
            self._logger.warning(msg)


# ------------------------------------------------------------------------------------------------ #
class ModelCheckpointRepo:
    """Repository for Model checkpoints

    Args:
        location (str): Directory containing model checkpoints.

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

    __location = "models/"

    def __init__(
        self,
        location: str = None,
        monitor: str = "val_loss",
        mode: str = "auto",
        save_weights_only: bool = False,
        save_best_only: bool = True,
        save_freq: str = "epoch",
        verbose: int = 1,
    ) -> None:
        self._location = location or self.__location
        self._monitor = monitor
        self._mode = mode
        self._save_weights_only = save_weights_only
        self._save_best_only = save_best_only
        self._save_freq = save_freq
        self._verbose = verbose

        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(level=logging.DEBUG)

    def get(self, name: str, stage: str) -> tf.keras.Model:
        """Returns the latest model for the give name and stage

        Args:
            name (str): Model name
            stage (str): Stage of the model.
        """

        directory = os.path.join(self._location, name, stage)
        pattern = name + "_" + stage + "*.keras"
        filepath = sorted(glob(directory + pattern, recursive=True))[-1]
        if len(filepath) == 0:
            msg = f"No model checkpoint found for {name}_{stage}"
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        model = tf.keras.models.load_model(filepath)
        epoch = self._get_epoch(filepath=filepath)
        return model, epoch

    def create_callback(
        self, name: str, stage: str
    ) -> tf.keras.callbacks.ModelCheckpoint:
        """Creates and returns the ModelCheckpoint callback.

        Args:
            name (str): The name of the model
            stage (str): Brief description of model stage.
        """
        filename = f"{name}_{stage}_"
        filename = filename + "{epoch:02d}-val_loss_{val_loss:.2f}.keras"
        filepath = os.path.join(self._location, name, stage, filename)
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

    def _get_epoch(self, filepath: str) -> int:
        return list(map(int, re.findall("\d+", filepath)))[0]
