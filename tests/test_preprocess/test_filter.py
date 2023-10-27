#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_preprocess/test_filter.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 23rd 2023 01:56:06 am                                                #
# Modified   : Thursday October 26th 2023 09:12:47 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from bcd.preprocess.filter import MeanFilter, MedianFilter, GaussianFilter, FilterParams
from bcd.preprocess.convert import ImageConverter, ImageConverterParams

FILTER_TASK_ID = "cbcf4c02-0e5f-46e5-b5f7-9a865d626cad"
CONVERTER_TASK_ID = "1fc60da3-2bcd-44dd-97b5-0cc6caa33f3e"
NUM_IMAGES = 15
FRAC_IMAGES = 0.005
STAGE = 1

PARAMS = FilterParams()
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.filter
class TestFilter:  # pragma: no cover
    # ============================================================================================ #
    def test_setup(self, case_ids, container, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Delete Existing Filter Images
        repo = container.repo.image()
        try:
            repo.delete_by_stage(uid=1)
        except Exception:
            pass

        # Add Stage 0 Images
        condition = lambda df: df["stage_uid"] == 0
        count = repo.count(condition=condition)
        if count == 0:
            params = ImageConverterParams(frac=FRAC_IMAGES)
            conv = ImageConverter(params=params, task_id=CONVERTER_TASK_ID)
            conv.execute()

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)

    # ============================================================================================ #
    def test_mean_filter(self, container, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # --------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        filter = MeanFilter(params=PARAMS, task_id=FILTER_TASK_ID)
        assert filter.stage_uid == STAGE
        assert filter.name == "MeanFilter"
        assert filter.task_id == FILTER_TASK_ID

        filter.execute()
        assert filter.images_processed == NUM_IMAGES
        condition = lambda df: df["stage_uid"] == 1
        assert repo.count(condition) == 15
        condition = lambda df: df["preprocessor"] == "MeanFilter"
        assert repo.count(condition) == 15

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
    def test_median_filter(self, container, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # --------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        filter = MedianFilter(params=PARAMS, task_id=FILTER_TASK_ID)
        assert filter.stage_uid == STAGE
        assert filter.name == "MedianFilter"
        assert filter.task_id == FILTER_TASK_ID

        filter.execute()
        assert filter.images_processed == NUM_IMAGES
        condition = lambda df: df["stage_uid"] == 1
        assert repo.count(condition) == 30
        condition = lambda df: df["preprocessor"] == "MedianFilter"
        assert repo.count(condition) == 15

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
    def test_gaussian_filter(self, container, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # --------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        filter = GaussianFilter(params=PARAMS, task_id=FILTER_TASK_ID)
        assert filter.stage_uid == STAGE
        assert filter.name == "GaussianFilter"
        assert filter.task_id == FILTER_TASK_ID

        filter.execute()
        assert filter.images_processed == NUM_IMAGES
        condition = lambda df: df["stage_uid"] == 1
        assert repo.count(condition) == 45
        condition = lambda df: df["preprocessor"] == "GaussianFilter"
        assert repo.count(condition) == 15

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
    def test_teardown(self, container, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        repo.delete_by_stage(uid=1)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)
