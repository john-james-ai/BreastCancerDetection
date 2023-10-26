#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_preprocess/test_preprocess_task.py                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 26th 2023 01:16:09 am                                              #
# Modified   : Thursday October 26th 2023 03:58:39 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from bcd.config import Config
from bcd.preprocess.convert import ImageConverter, ImageConverterParams, ImageConverterTask

TASK = None

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.task
class TestImageConverterTask:  # pragma: no cover
    # ============================================================================================ #
    def test_task(self, caplog):
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
        params = ImageConverterParams(frac=0.005)
        task = ImageConverterTask.create(application=ImageConverter, params=params, config=Config)
        assert task.id is not None
        assert task.name == "ImageConverter"
        assert task.application == ImageConverter
        assert isinstance(task.params, ImageConverterParams)
        assert task.mode == "test"
        assert task.stage_id == 0
        assert task.stage == "converted"
        assert task.module == "bcd.preprocess.convert"

        # Run task
        task.run()
        assert task.images_processed == 15
        assert isinstance(task.image_processing_time, float)
        assert isinstance(task.started, datetime)
        assert isinstance(task.ended, datetime)
        assert isinstance(task.duration, float)
        assert task.state == "SUCCESS"

        task = ImageConverterTask.from_df(df=task.as_df())
        assert task.id is not None
        assert task.name == "ImageConverter"
        assert task.application == ImageConverter
        assert isinstance(task.params, ImageConverterParams)
        assert task.mode == "test"
        assert task.stage_id == 0
        assert task.stage == "converted"
        assert task.images_processed == 15
        assert isinstance(task.image_processing_time, float)
        assert isinstance(task.started, datetime)
        assert isinstance(task.ended, datetime)
        assert isinstance(task.duration, float)
        assert task.state == "SUCCESS"

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
    def test_teardown(self, container, caplog):
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
        repo = container.repo.image()
        repo.delete_by_stage(stage_id=0)
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
