#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_data/test_explore/test_dataset.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 18th 2023 06:37:38 pm                                             #
# Modified   : Saturday October 21st 2023 02:06:42 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

import pandas as pd

from bcd.analyze.explore.case import CaseExplorer


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

CASES_FP = "tests/data/cooked/cases.csv"


@pytest.mark.cases
@pytest.mark.dataset
class TestDataset:  # pragma: no cover
    # ============================================================================================ #
    def test_model_data(self, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        ds = CaseExplorer(filepath=CASES_FP)
        X_train, y_train, X_test, y_test = ds.get_model_data()
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)

        XC_train, yc_train, XC_test, yc_test = ds.get_calc_model_data()
        assert X_train.shape[1] > XC_train.shape[1]
        assert X_test.shape[1] > XC_test.shape[1]
        assert isinstance(yc_train, pd.Series)
        assert isinstance(yc_test, pd.Series)

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
