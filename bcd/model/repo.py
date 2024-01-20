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
# Modified   : Friday January 19th 2024 03:00:02 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import os
import re
import shutil
from glob import glob

import tensorflow as tf


# ------------------------------------------------------------------------------------------------ #
class ModelRepo:
    """Repository for Model checkpoints

    Args:
        factory (type[ModelFactory]): A Model factory class type
    """

    __location = "models/"
    __history_filename = "_history.pkl"

    def __init__(
        self,
        location: str = None,
    ) -> None:
        self._location = location or self.__location
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(level=logging.DEBUG)

    def get(self, name: str, stage: str) -> tf.keras.Model:
        """Returns the model for the designated name and stage.

        Args:
            name (str): Model name
            stage (str): Stage of the model.
        """

        directory = os.path.join(self._location, name, stage)
        pattern = name + "_" + stage + "*.keras"
        try:
            filepath = sorted(glob(directory + "/" + pattern, recursive=True))[-1]
        except IndexError as exc:
            msg = f"No model checkpoint found for {name}_{stage} in {directory}."
            self._logger.exception(msg)
            raise FileNotFoundError(msg) from exc

        model = tf.keras.models.load_model(filepath)

        msg = f"Loaded {name}_{stage} model from the repository."
        self._logger.info(msg)

        return model

    def exists(self, name: str, stage: str) -> bool:
        """Determines whether models exist for the designated name and stage

        Args:
            name (str): Model name
            stage (str): Stage of the model.
        """
        directory = os.path.join(self._location, name, stage)
        if os.path.exists(directory):
            return len(os.listdir(directory)) > 0
        return False

    def remove(self, name: str, stage: str, confirm: bool = False) -> bool:
        """Removes model from the repository
        Args:
            name (str): Model name
            stage (str): Stage of the model.
        """
        directory = os.path.join(self._location, name, stage)
        if os.path.exists(directory):
            if confirm:
                msg = f"Confirm removal of {name} {stage} from the repository. [Y/N]"
                response = input(msg)
                if "y" in response.lower():
                    self._delete_models(name=name, stage=stage)
            else:
                self._delete_models(name=name, stage=stage)

    def _delete_models(self, name: str, stage: str) -> None:
        """Deletes models from the repository. Assumes directory exists."""
        directory = os.path.join(self._location, name, stage)
        try:
            n = len(os.listdir(directory))
            msg = f"Deleted {n} {name} models from stage {stage}."
            shutil.rmtree(directory)
            self._logger.info(msg)
        except FileNotFoundError:
            pass

    def create_callback(
        self,
        name: str,
        stage: str,
        monitor: str = "val_loss",
        mode: str = "auto",
        save_weights_only: bool = False,
        save_best_only: bool = True,
        save_freq: str = "epoch",
        verbose: int = 1,
    ) -> tf.keras.callbacks.ModelCheckpoint:
        """Creates and returns the ModelCheckpoint callback.

        Args:
            name (str): The name of the model stage (str): Brief description of model stage.
            monitor: The metric name to monitor. Typically the metrics are set by the Model.compile
            method. Note: Prefix the name with "val_" to monitor validation metrics. Use "loss" or
            "val_loss" to monitor the model's total loss. If you specify metrics as strings, like
            "accuracy", pass the same string (with or without the "val_" prefix). If you pass
            metrics.Metric objects, monitor should be set to metric.name If you're not sure about
            the metric names you can check the contents of the history.history dictionary returned
            by history = model.fit() Multi-output models set additional prefixes on the metric
            names.

            mode (str): one of {"auto", "min", "max"}. If save_best_only=True, the decision to
            overwrite the current save file is made based on either the maximization or the
            minimization of the monitored quantity. For val_acc, this should be "max", for val_loss
            this should be "min", etc. In "auto" mode, the mode is set to "max" if the quantities
            monitored are "acc" or start with "fmeasure" and are set to "min" for the rest of the
            quantities.

            save_weights_only (bool): if True, then only the model's weights will be saved
            (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).

            save_freq (str): "epoch" or integer. When using "epoch", the callback saves the model
            after each epoch. When using integer, the callback saves the model at end of this many
            batches. If the Model is compiled with steps_per_execution=N, then the saving criteria
            will be checked every Nth batch. Note that if the saving isn't aligned to epochs, the
            monitored metric may potentially be less reliable (it could reflect as little as 1
            batch, since the metrics get reset every epoch). Defaults to "epoch".

            verbose (int): Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages
            when the callback takes an action.
        """
        filename = f"{name}_{stage}_"
        filename = filename + "{epoch:02d}-val_loss_{val_loss:.2f}.keras"
        filepath = os.path.join(self._location, name, stage, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            mode=mode,
            save_weights_only=save_weights_only,
            save_best_only=save_best_only,
            save_freq=save_freq,
            verbose=verbose,
        )

    def _get_epoch(self, name: str, stage: str, filepath: str) -> int:
        filename = os.path.basename(filepath)
        name_stage = f"{name}_{stage}_"
        epoch_score = filename.replace(name_stage, "")
        return list(map(int, re.findall(r"\d+", epoch_score)))[0]
