#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_core/test_orchestration/test_task.py                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 26th 2023 09:56:35 am                                              #
# Modified   : Sunday October 29th 2023 03:59:05 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
from datetime import datetime

import pytest

from bcd.core.orchestration.task import Task
from bcd.preprocess.image.convert import ImageConverter, ImageConverterParams

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.task
class TestTask:  # pragma: no cover
    # ============================================================================================ #
    def test_task(self):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        params = ImageConverterParams(frac=0.005)
        task = Task.create(application=ImageConverter, params=params, mode="test")
        task.job_id = "some_job_id"
        assert task.uid is not None
        assert task.name == "ImageConverter"
        assert task.stage_id == 0
        assert task.stage == "Converted"
        assert task.application == ImageConverter
        assert task.application_name == "ImageConverter"
        assert task.application_module == "bcd.preprocess.image.convert"
        assert task.params == params
        assert isinstance(task.params_string, str)
        assert task.params_name == "ImageConverterParams"
        assert task.params_module == "bcd.preprocess.image.convert"

        task.run()
        assert task.images_processed == 15
        assert isinstance(task.image_processing_time, float)
        assert isinstance(task.started, datetime)
        assert isinstance(task.ended, datetime)
        assert isinstance(task.duration, float)
        assert task.state == "SUCCESS"
        assert task.job_id == "some_job_id"

        logging.debug(task)

        df = task.as_df()
        task2 = Task.from_df(df=df)
        assert task == task2

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_teardown(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        repo.delete_by_transformer(transformer="ImageConverter")

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
