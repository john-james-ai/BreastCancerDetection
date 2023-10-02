#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/selection.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 2nd 2023 06:55:34 am                                                 #
# Modified   : Monday October 2nd 2023 04:20:56 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import sys
import os
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

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
    def feature_importances(self) -> pd.DataFrame:
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

        if force or not os.path.exists(self._filepath):
            self._run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            self._extract_feature_importance()
            self.save(filepath=self._filepath)
        else:
            self.load()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts using the best model."""
        if self._best_model is None:
            msg = "Model Selector not run. "
            logger.exception(msg)
        else:
            return self._best_model.predict(X)

    def score(self, y_true, y_pred) -> None:
        print(accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))

    def save(self, filepath: str = None) -> None:
        self._filepath = filepath or self._filepath

        if self._best_model is not None:
            os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
            with open(self._filepath, "wb") as f:
                pickle.dump(self._best_model, f)
            msg = f"Saved {self._best_model_name} grid search  pipeline to file: {self._filepath}."
            print(msg)
        else:
            msg = "Model Selector not yet run. No pipeline to save."
            logger.exception(msg)

    def load(self) -> None:
        try:
            with open(self._filepath, "rb") as f:
                self._best_model = pickle.load(f)
        except FileNotFoundError:
            msg = f"No model found at {self._filepath}"
            logger.exception(msg)
            raise

    def _run(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
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
            # Best training data accuracy
            msg = f"Best Training Accuracy: {round(gs.best_score_,3)}"
            print(msg)
            # Predict on test data with best params
            y_pred = gs.predict(X_test)
            # Test data accuracy of model with best params
            msg = f"Test set accuracy score for best params: {round(accuracy_score(y_test, y_pred),3)}."
            print(msg)
            # Track best (highest test accuracy) model
            if accuracy_score(y_test, y_pred) > self._best_score:
                self._best_score = accuracy_score(y_test, y_pred)
                self._best_model = gs
                self._best_model_name = name
        msg = f"\nClassifier with best test set accuracy: {self._best_model_name}."
        print(msg)

    def _extract_feature_importance(self) -> None:
        try:
            feature_imp = self._best_model.best_estimator_.named_steps["clf"].coef_
            feature_imp = pd.DataFrame(data=feature_imp, columns=self._features).T.reset_index()
        except AttributeError:
            feature_imp = self._best_model.best_estimator_.named_steps["clf"].feature_importances_
            feature_imp = pd.DataFrame(data=feature_imp, index=self._features).reset_index()

        feature_imp.columns = ["Feature", "Coefficient"]
        # Add the absolute value of the coefficients for filtering and sorting
        feature_imp["abs"] = np.abs(feature_imp["Coefficient"])
        # Filter features with zero coefficients
        feature_imp = feature_imp.loc[feature_imp["abs"] > 0]
        # Sort importances by absolute value of the coefficient
        self._feature_importance = feature_imp.sort_values(by=["abs"], ascending=False)
