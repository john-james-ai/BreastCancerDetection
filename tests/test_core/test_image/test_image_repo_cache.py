#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_core/test_image/test_image_repo_cache.py                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 02:26:44 am                                                #
# Modified   : Friday October 27th 2023 02:10:54 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import inspect
from datetime import datetime
import pytest
import logging
import shutil

import pandas as pd

from bcd.core.image.entity import Image
from bcd.config import Config

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

IMAGE_FP = "tests/data/images"

@pytest.mark.repo
@pytest.mark.image_repo_cache
class TestImageRepo:  # pragma: no cover
    # ============================================================================================ #
    def test_setup(self, container):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        shutil.rmtree(IMAGE_FP, ignore_errors=True)
        repo = container.repo.image()
        try:
            repo.delete_by_mode()
        except Exception:
            pass
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)

    # ============================================================================================ #
    def test_add_cache(self, images, container):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        for image in images:
            repo.add(image=image)
            assert not os.path.exists(image.filepath)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)

    # ============================================================================================ #
    def test_save(self, images, container):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        for image in images:
            assert not os.path.exists(image.filepath)

        repo.save()
        for image in images:
            assert os.path.exists(image.filepath)
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
    def test_delete(self, images, container):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        repo.delete_by_mode()
        for image in images:
            assert not os.path.exists(image.filepath)
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

