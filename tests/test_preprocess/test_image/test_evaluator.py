#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_preprocess/test_image/test_evaluator.py                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday November 1st 2023 08:43:36 pm                                             #
# Modified   : Wednesday November 1st 2023 09:04:30 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
from datetime import datetime

import pytest

from bcd.preprocess.image.flow.builder import TaskBuilder
from bcd.preprocess.image.flow.convert import ConverterTask, ConverterTaskParams
from bcd.preprocess.image.flow.evaluate import EvaluatorTask, EvaluatorTaskParams
from bcd.preprocess.image.method.convert import Converter
from bcd.preprocess.image.method.denoise import FilterParams, GaussianFilter

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.evaluator
class TestEvaluatorTask:  # pragma: no cover
    # ============================================================================================ #
    def test_mode(self, current_mode):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        if current_mode != "test":
            msg = "\nCHANGE MODE TO TEST BEFORE RUNNING PYTEST!\nExiting pytest!\n"
            logger.exception(msg)
            pytest.exit(msg)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_setup(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        if repo.count() == 0:
            uow = container.dal.uow()
            task_params = ConverterTaskParams(frac=0.005, n_jobs=6)
            builder = TaskBuilder(uow=uow)
            builder.set_task(task=ConverterTask)
            builder.add_task_params(params=task_params)
            builder.add_method(method=Converter())
            task = builder.task
            task.run()
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_evaluator(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        uow = container.dal.uow()
        task_params = EvaluatorTaskParams(n=5, input_stage_id=0)
        method = GaussianFilter()
        method_params = FilterParams()
        builder = TaskBuilder(uow=uow)
        builder.set_task(task=EvaluatorTask)
        builder.add_task_params(params=task_params)
        builder.add_method(method=method)
        builder.add_method_params(params=method_params)
        task = builder.task
        task.run()

        assert uow.eval_repo.count() == 5
        evals = uow.eval_repo.get()
        logger.debug(evals)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_teardown(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.eval_repo()
        repo.delete_by_mode()
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
