#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/analyze/explore/multivariate/pipeline.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 2nd 2023 07:24:02 am                                                 #
# Modified   : Sunday November 5th 2023 09:19:45 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""PipelineBuilder Module"""
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bcd.analyze.explore.multivariate.base import BasePipelineBuilder

# ------------------------------------------------------------------------------------------------ #


class PipelineBuilder(BasePipelineBuilder):
    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self._scaler = None
        self._classifier = None
        self._params = None
        self._gridsearch_cv = None
        self._scorer = None
        self._jobs = None

    def reset(self) -> None:
        self._pipeline = None
        self._scaler = None
        self._classifier = None
        self._params = None
        self._gridsearch_cv = None
        self._scorer = None
        self._jobs = None

    @property
    def pipeline(self) -> Pipeline:
        """Returns a GridSearchCV Pipeline"""
        return self._gridsearch_cv

    def set_jobs(self, jobs: int = 6) -> None:
        self._jobs = jobs

    def set_standard_scaler(self) -> None:
        """Sets the standard scaler in the Pipeline"""
        self._scaler = ("scaler", StandardScaler())

    def set_classifier(self, classifier: BaseEstimator, params: list) -> None:
        """Sets the classifier and the parameters"""
        self._classifier = ("clf", classifier)
        self._params = params

    def set_scorer(self, scorer: str) -> None:
        self._scorer = scorer

    def build_gridsearch_cv(self, cv: int = 10, **kwargs) -> None:
        """Creates the GridSearchCV object"""
        self._pipeline = Pipeline([self._scaler, self._classifier])
        self._gridsearch_cv = GridSearchCV(
            estimator=self._pipeline,
            param_grid=self._params,
            scoring=self._scorer,
            cv=cv,
            n_jobs=self._jobs,
        )
