#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_model/test_model_selection.py                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 2nd 2023 08:22:42 am                                                 #
# Modified   : Monday November 6th 2023 12:16:14 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from bcd.analyze.explore.multivariate.pipeline import PipelineBuilder
from bcd.analyze.explore.multivariate.selection import ModelSelector

CASE_FP = os.path.abspath("data/meta/3_cooked/cases.csv")
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long, no-member, logging-format-interpolation
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

FEATURES = [
    "breast_density",
    "abnormality_id",
    "assessment",
    "subtlety",
    "AT_calcification",
    "AT_mass",
    "LR_LEFT",
    "LR_RIGHT",
    "IV_CC",
    "IV_MLO",
    "CT_AMORPHOUS",
    "CT_COARSE",
    "CT_DYSTROPHIC",
    "CT_EGGSHELL",
    "CT_FINE_LINEAR_BRANCHING",
    "CT_LARGE_RODLIKE",
    "CT_LUCENT_CENTERED",
    "CT_MILK_OF_CALCIUM",
    "CT_PLEOMORPHIC",
    "CT_PUNCTATE",
    "CT_ROUND_AND_REGULAR",
    "CT_SKIN",
    "CT_VASCULAR",
    "CD_CLUSTERED",
    "CD_LINEAR",
    "CD_REGIONAL",
    "CD_DIFFUSELY_SCATTERED",
    "CD_SEGMENTAL",
    "MS_IRREGULAR",
    "MS_ARCHITECTURAL_DISTORTION",
    "MS_OVAL",
    "MS_LYMPH_NODE",
    "MS_LOBULATED",
    "MS_FOCAL_ASYMMETRIC_DENSITY",
    "MS_ROUND",
    "MS_ASYMMETRIC_BREAST_TISSUE",
    "MM_SPICULATED",
    "MM_ILL_DEFINED",
    "MM_CIRCUMSCRIBED",
    "MM_OBSCURED",
    "MM_MICROLOBULATED",
]


@pytest.mark.model
class TestModelSelector:  # pragma: no cover
    # ============================================================================================ #
    def test_pipelinebuilder(self):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Load data
        cases = pd.read_csv(CASE_FP)
        train = cases.loc[cases["fileset"] == "train"]
        X_train = train.loc[:, train.columns != "cancer"][FEATURES]
        y_train = train["cancer"]

        test = cases.loc[cases["fileset"] == "test"]
        X_test = test.loc[:, test.columns != "cancer"][FEATURES]
        y_test = test["cancer"]
        X = pd.concat([X_train, X_test], axis=0)
        y = pd.concat([y_train, y_test], axis=0)

        # Test Pipeline Builder
        pb = PipelineBuilder()
        pb.set_jobs(6)
        pb.set_standard_scaler()
        # Add Logistic Regression
        c = [1.0, 0.5, 0.1]
        classifier = LogisticRegression(random_state=5)
        params = [{"clf__penalty": ["l1", "l2"], "clf__C": c, "clf__solver": ["liblinear"]}]
        pb.set_classifier(classifier=classifier, params=params)
        pb.set_scorer(scorer="accuracy")
        pb.build_gridsearch_cv()
        lr = pb.pipeline

        # Create SVC Pipeline
        pb.reset()
        pb.set_jobs(6)
        pb.set_standard_scaler()

        c = [1, 2, 3, 4, 5]
        clf = SVC(random_state=5)
        params = [{"clf__kernel": ["linear"], "clf__C": c}]
        pb.set_classifier(classifier=clf, params=params)
        pb.set_scorer("accuracy")
        pb.build_gridsearch_cv()
        svc = pb.pipeline

        assert isinstance(lr, GridSearchCV)
        assert isinstance(svc, GridSearchCV)

        # Test Selector
        selector = ModelSelector(filepath="tests/data/best_model.pkl")
        selector.add_pipeline(pipeline=lr, name="Logistic Regression")
        selector.add_pipeline(pipeline=svc, name="Support Vector Classifier")
        selector.run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, force=True)
        y_pred = selector.predict(X)
        selector.score(y_true=y, y_pred=y_pred)
        clf = selector.best_estimator
        assert isinstance(clf.coef_, np.ndarray)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
