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
# Modified   : Monday January 15th 2024 06:25:23 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import os
from datetime import datetime
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
        description: str,
        history: tf.keras.callbacks.History,
        monitor: str = "val_loss",
    ) -> None:
        """Adds a model to the repository

        Args:
            model (tf.keras.Model): Model to be saved to the repository.
            name (str): Name of the model
            description (str): Description of the model.
            history (tf.keras.callbacks.History): A history object returned from the fit method
            monitor (str): The metric that was monitored
        """
        dt = datetime.now()
        version = self._get_version(name=name)
        score, epoch = self._get_best_score(history=history, monitor=monitor)
        epochs = self._get_epochs(history=history)
        filepath = self._format_filepath(
            name=name,
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
            "description": description,
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
        return df["name"].loc[df["name"] == name].count() + 1

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
