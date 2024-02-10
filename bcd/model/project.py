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
# Modified   : Saturday February 10th 2024 10:41:17 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
# pylint: disable=wrong-import-order, protected-access, consider-iterating-dictionary
# ------------------------------------------------------------------------------------------------ #
"""Project Module"""
import itertools
import logging
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import wandb
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

from bcd.model.artifact import ModelArtifact
from bcd.model.experiment import Experiment
from bcd.model.factory_old import ModelFactory
from bcd.model.repo import ModelRepo
from bcd.utils.math import find_factors

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

    def __init__(
        self,
        name: str,
        config: dict,
        factory: ModelFactory,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        repo: ModelRepo,
        optimizer: tf.keras.optimizers,
        callbacks: list,
        metrics: list,
        force: bool,
    ) -> None:
        self._name = name
        self._config = config
        self._factory = factory
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._repo = repo
        self._optimizer = optimizer
        self._callbacks = callbacks
        self._metrics = metrics
        self._force = force

        self._run_started = None

        self._experiments = {}
        self._models = {}
        self._log = []

        self._entity = os.getenv("WANDB_ENTITY")
        self._version = factory.version
        self._dataset = config["dataset"]
        self._runs = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def log(self) -> pd.DataFrame:
        return pd.DataFrame(data=self._log)

    def run(self, names: list = None) -> None:
        """Run the named experiments.

        Args:
            names (list): List of model names to run. Optional. If None
                All models will be run..
        """
        names = names or self._factory.model_names
        for name in names:
            self.run_experiment(name=name)
        return self.log

    def run_experiment(self, name: str) -> None:
        """Runs the experiment

        Args:
            name (str): Model name (alias of the model object.)
        """
        self._start_experiment(name=name)
        model = self._factory.create_model(name=name)
        model.summary()

        self._config["run_name"] = name
        experiment = Experiment(
            model=model,
            config=self._config,
            repo=self._repo,
            optimizer=self._optimizer,
            callbacks=self._callbacks,
            metrics=self._metrics,
            force=self._force,
        )
        experiment.run(train_ds=self._train_ds, val_ds=self._val_ds)
        self._models[name] = experiment.model
        self._end_experiment(name)

    def classification_report(self, name: str, data: tf.data.Dataset = None) -> None:
        """Prints a classification report.

        Classification report is for the given dataset or the validation
        set if no data is None

        Args:
            name (str) The model name
            data (tf.data.Dataset) A dataset for which the classification report
                is to be rendered. If None, the validation set will be used.
        """
        data = data or self._val_ds

        model = self._get_model(name=name, ignore_errors=False)

        actual = np.concatenate([y for x, y in data], axis=0)
        predicted = (model.predict(data) > 0.5).astype("int32")

        print(classification_report(actual, predicted, target_names=data.class_names))

    def plot_confusion_matrix(
        self,
        name: str,
        data: tf.data.Dataset = None,
        ax: plt.Axes = None,
        normalize: bool = False,
    ) -> None:
        """Plots a confusion matrix for the validation set.

        Args:
            name (str) The model name
            data (tf.data.Dataset): Dataset for which the confusion matrix is
                to be computed. If None, the validation set will be used.
            ax (plt.Axes): Matplotlib Axes object. Optional
            normalize (bool): Whether to normalize the values to 1 across rows.

        """
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))

        data = data or self._val_ds

        model_id = ModelArtifact(
            name=name, version=self._version, dataset=self._dataset
        ).id

        model = self._repo.get(model_id=model_id)

        actual = np.concatenate([y for x, y in data], axis=0)
        predicted = (model.predict(data) > 0.5).astype("int32")

        cm = confusion_matrix(actual, predicted)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        cmap = "Blues"
        ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(f"{self._name}\nConfusion matrix", fontsize=9)

        tick_marks = np.arange(len(data.class_names))
        ax.set_xticks(tick_marks, data.class_names, rotation=90, fontsize=8)
        ax.set_yticks(tick_marks, data.class_names, fontsize=8)

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], ".2f"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

        ax.set_ylabel("True label", fontsize=9)
        ax.set_xlabel("Predicted label", fontsize=9)

    def plot_confusion_matrices(
        self, data: tf.data.Dataset = None, normalize: bool = False
    ) -> None:
        """Plots a confusion matrix for each experiment.

        Args:
             data (tf.data.Dataset): Dataset for which the confusion matrix is
                to be computed. If None, the validation set will be used.
            normalize (bool): Whether to normalize the values to 1 across rows.
        """
        data = data or self._val_ds

        # Dynamically format the axis geometry
        names = self._factory.model_names
        n = len(names)
        nrows, ncols = find_factors(c=n)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))

        for idx, ax in enumerate(axes.flat):
            self.plot_confusion_matrix(
                name=names[idx], data=data, ax=ax, normalize=normalize
            )

        fig.suptitle(f"{self._name} Confusion Matrices", fontsize=10)
        plt.tight_layout()

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

    def summarize(self) -> pd.DataFrame:
        """Returns all run summaries"""
        api = wandb.Api()
        runs = api.runs(self._entity + "/" + self._name)
        summary_list, name_list = [], []
        for run in runs:
            # Summary contains key/value pairs for run metrics
            summary_list.append(run.summary._json_dict)
            # Human-readable name of the run
            name_list.append(run.name)

        return pd.DataFrame({"name": name_list, "summary": summary_list})

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

    def _start_experiment(self, name: str) -> None:
        self._run_started = datetime.now()
        started = self._run_started.strftime("%Y-%m-%d %H:%M:%S")
        print("\n\n____________________________________________________________")
        print(f"                     Experiment: {name}")
        print(f"                    {started}")
        print("============================================================\n")

    def _end_experiment(self, name: str) -> None:
        log = {}
        log["name"] = name
        log["started"] = self._run_started
        log["ended"] = datetime.now()
        log["runtime"] = (log["ended"] - log["started"]).total_seconds()
        self._log.append(log)

        ended = log["ended"].strftime("%Y-%m-%d %H:%M:%S")
        runtime = round(log["duration"], 2)

        print("\n____________________________________________________________")
        print(f"     Experiment: {name} Completed at {ended}")
        print(f"                Runtime: {runtime} seconds")
        print("============================================================\n\n")

    def _get_model(self, name: str, ignore_errors: bool = True) -> tf.keras.Model:
        """Retrieves a model by name.

        This method attempts to obtain the model from the experiment in memory.
        If not found, it checks the model_cache in case the model has already
        been loaded. If the experiment nor the model exists in memory,
        the repository is checked.
        """
        # Check the experiment cache.
        if name in self._experiments.keys():
            return self._experiments[name].model
        # Check the model cache
        elif name in self._models.keys():
            return self._models[name]
        else:
            model_id = ModelArtifact(
                name=name, version=self._version, dataset=self._dataset
            ).id
            try:
                return self._repo.get(model_id=model_id)
            except FileNotFoundError:
                if ignore_errors:
                    pass
                else:
                    raise
