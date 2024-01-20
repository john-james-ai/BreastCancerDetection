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
# Modified   : Friday January 19th 2024 12:25:51 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Model Visualizer Module"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")

warnings.simplefilter(action="ignore", category=FutureWarning)


# ------------------------------------------------------------------------------------------------ #
class X4LearningVisualizer:
    """Visualizes learning curves throughout the transfer learning process."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._scores = {}

    def plot_learning_curves(self) -> None:
        """Plots learning curve for all epochs in the history.

        Plots learning curve. If mode is 'a', the history is appended
        to the existing history and the plot will cover all epochs
        in the cumulative history. If mode is 'w', only the scores
        from the history parameter are plotted.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        epochs_range = np.arange(start=1, stop=len(self._scores["loss"]) + 1)

        d_train_loss = {
            "epoch": epochs_range,
            "dataset": "train",
            "loss": self._scores["loss"],
        }
        d_val_loss = {
            "epoch": epochs_range,
            "dataset": "validation",
            "loss": self._scores["val_loss"],
        }
        d_train_accuracy = {
            "epoch": epochs_range,
            "dataset": "train",
            "accuracy": self._scores["accuracy"],
        }
        d_val_accuracy = {
            "epoch": epochs_range,
            "dataset": "validation",
            "accuracy": self._scores["val_accuracy"],
        }

        df_train_loss = pd.DataFrame(data=d_train_loss)
        df_val_loss = pd.DataFrame(data=d_val_loss)
        df_loss = pd.concat([df_train_loss, df_val_loss], axis=0)

        df_train_accuracy = pd.DataFrame(data=d_train_accuracy)
        df_val_accuracy = pd.DataFrame(data=d_val_accuracy)
        df_accuracy = pd.concat([df_train_accuracy, df_val_accuracy], axis=0)

        sns.lineplot(data=df_loss, x="epoch", y="loss", hue="dataset", ax=axes[0])
        title = f"{self._name}\nTraining and Validation Loss"
        axes[0].set_title(title)

        sns.lineplot(
            data=df_accuracy, x="epoch", y="accuracy", hue="dataset", ax=axes[1]
        )
        title = f"{self._name}\nTraining and Validation Accuracy"
        axes[1].set_title(title)

        fig.suptitle(f"{self._name} Learning Curves")
        plt.tight_layout()
        plt.show()

    def add_scores(self, history: tf.keras.callbacks.History, mode: str = "a") -> None:
        """Adds scores to the visualizer

        Args:
            history (tf.keras.callbacks.History): History object
            mode (str): Whether to append ('a') or overwrite ('w').
        """
        if mode == "a":
            for metric, scores in history.history.items():
                try:
                    self._scores[metric].extend(scores)
                except KeyError:
                    self._scores[metric] = scores
        else:
            for metric, scores in history.history.items():
                self._scores[metric] = scores
