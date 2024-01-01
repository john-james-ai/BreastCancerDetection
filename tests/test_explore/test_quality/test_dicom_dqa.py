#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_explore/test_quality/test_dicom_dqa.py                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 06:38:22 am                                              #
# Modified   : Thursday December 28th 2023 09:29:37 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
from datetime import datetime

import pandas as pd
import pytest

from bcd.analyze.dqa.dicom import DicomDQA

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long, no-member, logging-format-interpolation
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

DICOM_FP = "data/meta/1_staged/dicom.csv"


@pytest.mark.dqa
@pytest.mark.dicomdqa
class TestDicomDQA:  # pragma: no cover
    # ============================================================================================ #
    def test_completeness(self):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        dqa = DicomDQA(filepath=DICOM_FP)
        result = dqa.analyze_completeness()

        assert isinstance(result.detail, pd.DataFrame)
        logger.debug(f"Completeness Summary\n{result.summary}")
        logger.debug(f"Completeness Detail\n{result.detail}")

        # Complete Rows
        df = dqa.get_complete_data()
        assert df.shape[0] == result.summary.complete_records

        # Incomplete Rows
        df = dqa.get_incomplete_data()
        assert df.shape[0] == result.summary.records - result.summary.complete_records

        # Incomplete rows by mass shape
        df = dqa.get_incomplete_data()
        assert df.shape[0] == 0

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
    def test_unique(self):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        dqa = DicomDQA(filepath=DICOM_FP)
        result = dqa.analyze_uniqueness()

        assert isinstance(result.detail, pd.DataFrame)
        logger.debug(f"Uniqueness Summary\n{result.summary}")
        logger.debug(f"Uniqueness Detail\n{result.detail}")

        df = dqa.get_unique_data()
        assert df.shape[0] == result.summary.unique_records

        df = dqa.get_duplicate_data()
        assert df.shape[0] == 0
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_validity(self):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        dqa = DicomDQA(filepath=DICOM_FP)
        result = dqa.analyze_validity()

        assert isinstance(result.detail, pd.DataFrame)
        logger.debug(f"Validity Summary\n{result.summary}")
        logger.debug(f"Validity Detail\n{result.detail}")

        df = dqa.get_valid_data()
        assert df.shape[0] == result.summary.valid_records

        df = dqa.get_invalid_data()
        assert df.shape[0] == result.summary.records - result.summary.valid_records
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
