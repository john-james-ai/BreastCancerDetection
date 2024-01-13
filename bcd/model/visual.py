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
# Modified   : Saturday January 13th 2024 01:01:00 pm                                              #
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


def learning_curve(model_name: str, history: tf.keras.callbacks.History) -> None:
    """Visualizes training and validation loss and accuracy."""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    train_accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(train_accuracy))

    d_train_loss = {"epoch": epochs_range, "dataset": "train", "loss": train_loss}
    d_val_loss = {"epoch": epochs_range, "dataset": "validation", "loss": val_loss}
    d_train_accuracy = {
        "epoch": epochs_range,
        "dataset": "train",
        "accuracy": train_accuracy,
    }
    d_val_accuracy = {
        "epoch": epochs_range,
        "dataset": "validation",
        "accuracy": val_accuracy,
    }

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

    fig.suptitle(f"{model_name} Performance ")
    plt.tight_layout()
