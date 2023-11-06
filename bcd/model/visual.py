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
# Modified   : Sunday November 5th 2023 06:51:46 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Model Visualizer Module"""
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")

warnings.simplefilter(action="ignore", category=FutureWarning)


# ------------------------------------------------------------------------------------------------ #
class ModelVisualizer:
    """Provides visualization of model training and validation performance."""

    def __init__(self, model_name: str, history) -> None:
        self._model_name = model_name
        self._history = history

    def visualize_training(self) -> None:
        """Visualizes training and validation loss and accuracy."""
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        train_accuracy = self._history.history["accuracy"]
        val_accuracy = self._history.history["val_accuracy"]

        train_loss = self._history.history["loss"]
        val_loss = self._history.history["val_loss"]

        epochs_range = range(len(train_accuracy))

        d_train_loss = {"epoch": epochs_range, "dataset": "train", "loss": train_loss}
        d_val_loss = {"epoch": epochs_range, "dataset": "validation", "loss": val_loss}
        d_train_accuracy = {"epoch": epochs_range, "dataset": "train", "accuracy": train_accuracy}
        d_val_accuracy = {"epoch": epochs_range, "dataset": "validation", "accuracy": val_accuracy}

        df_train_loss = pd.DataFrame(data=d_train_loss)
        df_val_loss = pd.DataFrame(data=d_val_loss)
        df_loss = pd.concat([df_train_loss, df_val_loss], axis=0)

        df_train_accuracy = pd.DataFrame(data=d_train_accuracy)
        df_val_accuracy = pd.DataFrame(data=d_val_accuracy)
        df_accuracy = pd.concat([df_train_accuracy, df_val_accuracy], axis=0)

        sns.lineplot(data=df_loss, x="epoch", y="loss", hue="dataset", ax=axes[0])
        axes[0].set_title("Training and Validation Loss")

        sns.lineplot(data=df_accuracy, x="epoch", y="accuracy", hue="dataset", ax=axes[1])
        axes[1].set_title("Training and Validation Accuracy")

        fig.suptitle(f"{self._model_name} Performance ")
        plt.tight_layout()
