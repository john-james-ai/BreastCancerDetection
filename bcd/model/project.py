#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/project.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday February 8th 2024 02:51:58 am                                              #
# Modified   : Thursday February 8th 2024 01:13:03 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
# pylint: disable=wrong-import-order
# ------------------------------------------------------------------------------------------------ #
"""Project Module"""
import logging
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from dotenv import load_dotenv

# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")
warnings.simplefilter(action="ignore", category=FutureWarning)
# ------------------------------------------------------------------------------------------------ #


class Project:
    """Project encapsulates a series of experiments for visualization and analysis.

    Args:
        name (str): Name of the project on Weights and Biases

    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entity = os.getenv("WANDB_ENTITY")
        self._runs = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def plot_learning_curve(self, name: str, ax: plt.Axes = None) -> None:
        """Plots the learning curve for a run.

        Args:
            name (str): The run mame
            ax (plt.Axes): Matplotlib Axes object. Optional.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        run = self._get_run(name=name)

        history = run.history()
        loss = history[["_step", "epoch/loss", "epoch/val_loss"]]
        loss.columns = ["epoch", "train", "validation"]
        loss = loss.melt(
            id_vars=["epoch"],
            value_vars=["train", "validation"],
            var_name="dataset",
            value_name="loss",
        )
        loss["epoch"] += 1

        sns.lineplot(data=loss, x="epoch", y="loss", hue="dataset", ax=ax)
        title = f"{name}\nLearning Curve"
        ax.set_title(title)
        plt.tight_layout()

    def plot_validation_curve(self, name: str, ax: plt.Axes = None) -> None:
        """Plots the learning curve for a run.

        Args:
            name (str): The run mame
            ax (plt.Axes): Matplotlib Axes object. Optional.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        run = self._get_run(name=name)

        history = run.history()
        loss = history[["_step", "epoch/accuracy", "epoch/val_accuracy"]]
        loss.columns = ["epoch", "train", "validation"]
        loss = loss.melt(
            id_vars=["epoch"],
            value_vars=["train", "validation"],
            var_name="dataset",
            value_name="accuracy",
        )
        loss["epoch"] += 1

        sns.lineplot(data=loss, x="epoch", y="accuracy", hue="dataset", ax=ax)
        title = f"{name}\nValidation Curve"
        ax.set_title(title)
        plt.tight_layout()

    def plot_performance_curves(self, name: str) -> None:
        """Plots the learning and validation curves.

        Args:
            name (str): The run mame
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        self.plot_learning_curve(name=name, ax=axes[0])
        self.plot_validation_curve(name=name, ax=axes[1])
        fig.suptitle(f"{name}\nPerformance Curves")

    def plot_model_performance(self, metric: str, ax: plt.Axes = None) -> None:
        """Plots model performance for the designated metric

        Args:
            metric (str): A metric from the Weights & Biases run summary.
            ax (plt.Axes)

        """
        self._export_runs()

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 12))

        names = []
        scores = []
        for run in self._runs:
            names.append(run.name)
            scores.append(run.summary[metric["metric"]])

        performance = pd.DataFrame({"model": names, metric["label"]: scores})

        sns.barplot(data=performance, x="model", y=metric["label"], ax=ax)
        ax.set_title(metric["label"])
        ax.set_xticklabels(names, rotation=90)

    def compare_models(self) -> None:
        """Plots model performance by metric on validation set.

        Validation accuracy, AUC, precision and recall metrics
        for each model is plotted.
        """
        metrics = [
            {"metric": "epoch/val_loss", "label": "Validation Loss"},
            {"metric": "epoch/val_accuracy", "label": "Validation Accuracy"},
            {"metric": "epoch/val_auc", "label": "Validation AUC"},
            {"metric": "epoch/val_precision", "label": "Validation Precision"},
            {"metric": "epoch/val_recall", "label": "Validation Recall"},
            {"metric": "_runtime", "label": "Runtime"},
        ]

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
        for idx, ax in enumerate(axes.flat):
            self.plot_model_performance(metric=metrics[idx], ax=ax)

        fig.suptitle("Model Performance", fontsize=14)
        plt.tight_layout()

    def summarize_run(self, name: str) -> dict:
        """Returns the run summary for the designated run

        Args:
            name (str): The run name
        """
        run = self._get_run(name=name)
        return run.summary

    def _get_run(self, name: str) -> wandb.apis.public.Run:
        self._export_runs()
        api = wandb.Api()
        for run in self._runs:
            if name.lower() == run.name.lower():
                return api.run(f"{self._entity}/{self._name}/{run.id}")

        msg = f"Run {name} not found."
        self._logger.exception(msg)
        raise FileNotFoundError(msg)

    def _export_runs(self) -> None:
        if self._runs is None:
            api = wandb.Api()
            self._runs = api.runs(self._entity + "/" + self._name)
