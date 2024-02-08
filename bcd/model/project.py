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
# Modified   : Thursday February 8th 2024 04:53:12 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
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
        """Plots the learning curve for a a run.

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

    def _get_run(self, name: str) -> wandb.apis.public.Run:
        self._export_runs()
        for run in self._runs:
            if name.lower() == run.name.lower():
                return run
        msg = f"Run {name} not found."
        self._logger.exception(msg)
        raise FileNotFoundError(msg)

    def _export_runs(self) -> None:
        if self._runs is None:
            api = wandb.Api()
            self._runs = api.runs(self._entity + "/" + self._name)
