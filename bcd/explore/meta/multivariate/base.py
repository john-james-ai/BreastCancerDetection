#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/analyze/explore/multivariate/base.py                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 2nd 2023 07:08:15 am                                                 #
# Modified   : Sunday November 5th 2023 09:16:18 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------------------------------------------ #
class BaseModelSelector(ABC):
    """Abstract base class for Model Selector classes."""

    @property
    @abstractmethod
    def best_model(self) -> BaseEstimator:
        """Returns the best model."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the model selector."""

    @abstractmethod
    def set_jobs(self, jobs: int = 6) -> None:
        """Set scoring function."""

    @abstractmethod
    def add_pipeline(self, pipeline: Pipeline, *args, **kwargs) -> None:
        """Adds a pipeline to the selector"""

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """Run the pipeline"""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Renders predictions using best model."""

    @abstractmethod
    def score(self, y_true, y_pred) -> None:
        """Renders accuracy and classification report."""

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Saves the best model to the designated filepath"""

    @abstractmethod
    def load(self) -> None:
        """Loads the best model to the designated filepath"""


# ------------------------------------------------------------------------------------------------ #


class BasePipelineBuilder(ABC):
    """Abstract base class for multivariate analysis pipeline builders."""

    @property
    @abstractmethod
    def pipeline(self) -> Pipeline:
        """Returns a GridSearchCV Pipeline"""

    @abstractmethod
    def set_standard_scaler(self) -> None:
        """Sets the standard scaler in the Pipeline"""

    @abstractmethod
    def set_classifier(self, *args, **kwargs) -> None:
        """Sets the classifier and the parameters"""

    @abstractmethod
    def set_scorer(self, scorer: str) -> None:
        """Set scoring function."""

    @abstractmethod
    def build_gridsearch_cv(self, *args, **kwargs) -> None:
        """Creates the GridSearchCV object"""
