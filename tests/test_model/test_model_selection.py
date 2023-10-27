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
# Modified   : Thursday October 26th 2023 08:05:56 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import inspect
from datetime import datetime
import pytest
import logging

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from bcd.model.selection import ModelSelector
from bcd.model.pipeline import PipelineBuilder

CALC_TRAIN_FP = os.path.abspath("data/cooked/calc_train.csv")
CALC_TEST_FP = os.path.abspath("data/cooked/calc_test.csv")
MASS_TRAIN_FP = os.path.abspath("data/cooked/mass_train.csv")
MASS_TEST_FP = os.path.abspath("data/cooked/mass_test.csv")

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.model
class TestModelSelector:  # pragma: no cover
    # ============================================================================================ #
    def test_pipelinebuilder(self, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Load data
        train = pd.read_csv(CALC_TRAIN_FP)
        X_train = train.loc[:, train.columns != "cancer"]
        y_train = train["cancer"]

        test = pd.read_csv(CALC_TEST_FP)
        X_test = test.loc[:, test.columns != "cancer"]
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
        pb.create_gridsearch_cv()
        lr = pb.pipeline

        # Create SVC Pipeline
        pb.reset()
        pb.set_jobs(6)
        pb.set_standard_scaler()

        c = [1, 2, 3, 4, 5]
        clf = SVC(random_state=5)
        params = [{"clf__kernel": ["linear", "rbf"], "clf__C": c}]
        pb.set_classifier(classifier=clf, params=params)
        pb.set_scorer("accuracy")
        pb.create_gridsearch_cv()
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
