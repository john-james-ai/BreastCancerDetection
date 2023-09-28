#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_data/test_case/test_case_dqa.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 06:38:22 am                                              #
# Modified   : Tuesday September 26th 2023 05:31:54 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

import pandas as pd

from bcd.data.dqa import DQASummary
from bcd.data.dqa.case import MassCaseDQA, CalcCaseDQA

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

CALC_CASES = "data/staged/calc_cases.csv"
MASS_CASES = "data/staged/mass_cases.csv"


@pytest.mark.dqa
@pytest.mark.casedqa
class TestCaseDQA:  # pragma: no cover
    # ============================================================================================ #
    def test_completeness(self, caplog):
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
        dqa = MassCaseDQA(filepath=MASS_CASES)
        result = dqa.analyze_completeness()

        assert isinstance(result.summary, DQASummary)
        assert isinstance(result.detail, pd.DataFrame)
        logger.debug(f"Completeness Summary\n{result.summary}")
        logger.debug(f"Completeness Detail\n{result.detail}")

        # Complete Rows
        df = dqa.get_complete_data()
        assert df.shape[0] == result.summary.rows_complete

        # Incomplete Rows
        df = dqa.get_incomplete_data()
        assert df.shape[0] == result.summary.rows - result.summary.rows_complete

        # Incomplete rows by mass shape
        df = dqa.get_incomplete_data(subset="mass_shape")
        assert df.shape[0] == 4

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

    # ============================================================================================ #
    def test_unique(self, caplog):
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
        dqa = MassCaseDQA(filepath=MASS_CASES)
        result = dqa.analyze_uniqueness()

        assert isinstance(result.summary, DQASummary)
        assert isinstance(result.detail, pd.DataFrame)
        logger.debug(f"Uniqueness Summary\n{result.summary}")
        logger.debug(f"Uniqueness Detail\n{result.detail}")

        df = dqa.get_unique_data()
        assert df.shape[0] == 1696

        df = dqa.get_duplicate_data()
        assert df.shape[0] == 0
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_mass_validity(self, caplog):
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
        dqa = MassCaseDQA(filepath=MASS_CASES)
        result = dqa.analyze_validity()

        assert isinstance(result.summary, DQASummary)
        assert isinstance(result.detail, pd.DataFrame)
        logger.debug(f"Validity Summary\n{result.summary}")
        logger.debug(f"Validity Detail\n{result.detail}")

        df = dqa.get_valid_data()
        assert df.shape[0] == 1478

        df = dqa.get_invalid_data()
        assert df.shape[0] == 1696 - 1478
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_calc_validity(self, caplog):
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
        dqa = CalcCaseDQA(filepath=CALC_CASES)
        result = dqa.analyze_validity()

        assert isinstance(result.summary, DQASummary)
        assert isinstance(result.detail, pd.DataFrame)
        logger.debug(f"Validity Summary\n{result.summary}")
        logger.debug(f"Validity Detail\n{result.detail}")

        df = dqa.get_valid_data()
        assert df.shape[0] == 1331

        df = dqa.get_invalid_data()
        assert df.shape[0] == 1872 - 1331
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
