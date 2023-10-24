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
# Modified   : Monday October 23rd 2023 06:02:35 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from bcd.preprocess.filter import (
    MeanFilter,
    MedianFilter,
    GaussianFilter,
)

FILEPATH = "tests/data/images"

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
        # Delete Tasks
        repo = container.repo.image()
        condition = lambda df: df["mode"] == "test"  # noqa
        repo.delete(condition=condition)

        # Add stage 0 images
        factory = container.repo.factory()
        for case_id in case_ids:
            # Obtain image
            image = factory.from_case(
                case_id=case_id, stage_id=0, task="TestFilter", taskrun_id="some_taskrun_id"
            )
            # Add image to repository
            repo.add(image=image)
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
    def test_mean_filter(self, caplog):
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
        filter = MeanFilter()
        filter.execute()
        taskrun = filter.taskrun

        logger.debug(taskrun)
        assert taskrun.task == "MeanFilter"
        assert taskrun.mode == "test"
        assert taskrun.stage_id == 1
        assert taskrun.stage == "denoise"
        assert isinstance(taskrun.started, datetime)
        assert isinstance(taskrun.ended, datetime)
        assert isinstance(taskrun.duration, float)
        assert isinstance(taskrun.images_processed, int)
        assert isinstance(taskrun.image_processing_time, float)
        assert taskrun.images_processed == 10
        assert taskrun.success is True

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
    def test_median_filter(self, caplog):
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
        filter = MedianFilter()
        filter.execute()
        taskrun = filter.taskrun

        logger.debug(taskrun)
        assert taskrun.task == "MedianFilter"
        assert taskrun.mode == "test"
        assert taskrun.stage_id == 1
        assert taskrun.stage == "denoise"
        assert isinstance(taskrun.started, datetime)
        assert isinstance(taskrun.ended, datetime)
        assert isinstance(taskrun.duration, float)
        assert isinstance(taskrun.images_processed, int)
        assert isinstance(taskrun.image_processing_time, float)
        assert taskrun.images_processed == 10
        assert taskrun.success is True

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
    def test_gaussian_filter(self, caplog):
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
        filter = GaussianFilter()
        filter.execute()
        taskrun = filter.taskrun

        logger.debug(taskrun)
        assert taskrun.task == "GaussianFilter"
        assert taskrun.mode == "test"
        assert taskrun.stage_id == 1
        assert taskrun.stage == "denoise"
        assert isinstance(taskrun.started, datetime)
        assert isinstance(taskrun.ended, datetime)
        assert isinstance(taskrun.duration, float)
        assert isinstance(taskrun.images_processed, int)
        assert isinstance(taskrun.image_processing_time, float)
        assert taskrun.images_processed == 10
        assert taskrun.success is True

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
