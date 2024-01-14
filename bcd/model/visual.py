#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /bcd/model/visual.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday November 5th 2023 02:56:04 pm                                                #
# Modified   : Sunday January 14th 2024 07:07:05 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Model Visualizer Module"""
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")

warnings.simplefilter(action="ignore", category=FutureWarning)


# ------------------------------------------------------------------------------------------------ #
class X4LearningVisualizer:
    """Visualizes learning curves throughout the transfer learning process."""

    def __init__(self, name) -> None:
        self._name = name
        self._train_accuracy = []
        self._train_loss = []
        self._val_accuracy = []
        self._val_loss = []

    def __call__(self, history: tf.keras.callbacks.History, mode: str = "a") -> None:
        """Extracts and plots cumulative metric data through the transfer learning process.

        Plots learning curve. Plots can be cumulative whereby the scores
        are added on each call, extending the number of epochs for which
        the scores are plotted if the mode is 'a' for append. Otherwise
        the scores are overridden on each call.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        self._update_scores(history=history, mode=mode)

        epochs_range = range(len(self._train_accuracy))

        d_train_loss = {
            "epoch": epochs_range,
            "dataset": "train",
            "loss": self._train_loss,
        }
        d_val_loss = {
            "epoch": epochs_range,
            "dataset": "validation",
            "loss": self._val_loss,
        }
        d_train_accuracy = {
            "epoch": epochs_range,
            "dataset": "train",
            "accuracy": self._train_accuracy,
        }
        d_val_accuracy = {
            "epoch": epochs_range,
            "dataset": "validation",
            "accuracy": self._val_accuracy,
        }

        df_train_loss = pd.DataFrame(data=d_train_loss)
        df_val_loss = pd.DataFrame(data=d_val_loss)
        df_loss = pd.concat([df_train_loss, df_val_loss], axis=0)

        df_train_accuracy = pd.DataFrame(data=d_train_accuracy)
        df_val_accuracy = pd.DataFrame(data=d_val_accuracy)
        df_accuracy = pd.concat([df_train_accuracy, df_val_accuracy], axis=0)

        sns.lineplot(data=df_loss, x="epoch", y="loss", hue="dataset", ax=axes[0])
        axes[0].set_title("Training and Validation Loss")

        sns.lineplot(
            data=df_accuracy, x="epoch", y="accuracy", hue="dataset", ax=axes[1]
        )
        axes[1].set_title("Training and Validation Accuracy")

        fig.suptitle(f"{self._name} Performance ")
        plt.tight_layout()

    def _update_scores(
        self, history: tf.keras.callbacks.History, mode: str = "a"
    ) -> None:
        """Appends or overwrites the scores to be plotted."""

        if mode == "a":
            self._train_accuracy.extend(history.history["accuracy"])
            self._val_accuracy.extend(history.history["val_accuracy"])

            self._train_loss.extend(history.history["loss"])
            self._val_loss.extend(history.history["val_loss"])
        else:
            self._train_accuracy = history.history["accuracy"]
            self._val_accuracy = history.history["val_accuracy"]

            self._train_loss = history.history["loss"]
            self._val_loss = history.history["val_loss"]
