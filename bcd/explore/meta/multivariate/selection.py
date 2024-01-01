#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/explore/meta/multivariate/selection.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 2nd 2023 06:55:34 am                                                 #
# Modified   : Saturday December 23rd 2023 09:36:26 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import GridSearchCV

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")
sns.set_palette("Blues_r")
# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class ModelSelector:
    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        self._pipelines = {}
        self._best_model = None
        self._best_score = 0
        self._best_classification_report = None
        self._best_model_name = None
        self._jobs = None
        self._features = None
        self._feature_importance = None

    @property
    def best_model(self) -> GridSearchCV:
        return self._best_model

    @property
    def best_estimator(self) -> BaseEstimator:
        return self._best_model.best_estimator_.named_steps["clf"]

    @property
    def feature_importance(self) -> pd.DataFrame:
        return self._feature_importance

    def add_pipeline(self, pipeline: GridSearchCV, name: str) -> None:
        """Adds a logistic regression model to the pipeline"""
        self._pipelines[name] = pipeline

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        force: bool = False,
    ) -> None:
        try:
            self._features = X_train.columns
        except AttributeError:
            pass

        if os.path.exists(self._filepath) and not force:
            self.load()
            msg = f"Best Model: {self._best_model_name} loaded from file."
            print(msg)
        else:
            self._run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            self.save(filepath=self._filepath)

        self._extract_feature_importance()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts using the best model."""
        if self._best_model is None:
            msg = "Model Selector not run. "
            logger.exception(msg)
        else:
            return self._best_model.predict(X)

    def score(self, y_true, y_pred) -> None:
        recall = recall_score(y_true, y_pred)
        msg = f"\n\t\tAccuracy of {self._best_model_name}: {round(recall,2)}"
        print(msg)
        msg = "\t\t\tClassification Report"
        print(msg)
        print(classification_report(y_true, y_pred))

    def save(self, filepath: str = None) -> None:
        self._filepath = filepath or self._filepath

        if self._best_model is not None:
            d = {
                "name": self._best_model_name,
                "score": self._best_score,
                "report": self._best_classification_report,
                "model": self._best_model,
            }

            os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
            with open(self._filepath, "wb") as f:
                pickle.dump(d, f)
            msg = f"Saved {self._best_model_name} grid search  pipeline to file: {os.path.relpath(self._filepath)}."
            print(msg)
        else:
            msg = "Model Selector not yet run. No pipeline to save."
            logger.exception(msg)

    def load(self) -> None:
        try:
            with open(self._filepath, "rb") as f:
                d = pickle.load(f)
                self._best_model_name = d["name"]
                self._best_score = d["score"]
                self._best_classification_report = d["report"]
                self._best_model = d["model"]

        except FileNotFoundError:
            msg = f"No model found at {self._filepath}"
            logger.exception(msg)
            raise

    def plot_feature_importance(
        self, title: str = None, ax: plt.Axes = None, palette: str = "Blues_r", **kwargs
    ) -> None:
        ax = sns.barplot(
            data=self._feature_importance,
            x="Importance",
            y="Feature",
            ax=ax,
            palette=palette,
            **kwargs,
        )
        if title is not None:
            ax.set_title(title)

        return ax

    def _run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        self._best_score = 0
        self._best_model_name = None
        self._best_model = None

        for name, gs in self._pipelines.items():
            msg = f"\nEstimator: {name}"
            print(msg)
            # Fit grid search
            gs.fit(X_train, y_train)
            # Best params
            msg = f"Best Params: {gs.best_params_}"
            print(msg)
            # Best training data recall
            msg = f"Best Training Recall: {round(gs.best_score_,3)}"
            print(msg)
            # Predict on test data with best params
            y_pred = gs.predict(X_test)
            # Test data recall of model with best params
            msg = f"Test set recall score for best params: {round(recall_score(y_test, y_pred),3)}."
            print(msg)
            # Capture recall and classification report for best model.
            if recall_score(y_test, y_pred) > self._best_score:
                self._best_score = recall_score(y_test, y_pred)
                self._best_classification_report = classification_report(
                    y_true=y_test, y_pred=y_pred
                )
                self._best_model = gs
                self._best_model_name = name
        msg = f"\nClassifier with best test set recall: {self._best_model_name}.\n"
        print(msg)

    def _extract_feature_importance(self) -> None:
        try:
            feature_imp = self._best_model.best_estimator_.named_steps["clf"].coef_

            feature_imp = pd.DataFrame(
                data=feature_imp, columns=self._features
            ).T.reset_index()
        except AttributeError:
            feature_imp = self._best_model.best_estimator_.named_steps[
                "clf"
            ].feature_importances_

            feature_imp = pd.DataFrame(
                data=feature_imp, index=self._features
            ).reset_index()

        feature_imp.columns = ["Feature", "Importance"]
        # Add the absolute value of the coefficients for filtering and sorting
        feature_imp["abs"] = np.abs(feature_imp["Importance"])
        # Filter features with zero coefficients
        feature_imp = feature_imp.loc[feature_imp["abs"] > 0]
        # Sort importances by absolute value of the coefficient
        self._feature_importance = feature_imp.sort_values(by=["abs"], ascending=False)
