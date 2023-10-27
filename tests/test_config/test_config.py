#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/test_config.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 04:08:08 pm                                             #
# Modified   : Thursday October 26th 2023 08:08:25 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from bcd.config import Config


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.config
class TestConfig:  # pragma: no cover
    """Test Configuration"""
    # ============================================================================================ #
    def test_get_mode(self):
        """Testing Get Mode"""
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        config = Config()
        assert config.mode == "test"

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)

    # ============================================================================================ #
    def test_set_mode(self, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        config = Config()
        config.mode = "dev"
        assert config.mode == "dev"

        with pytest.raises(ValueError):
            config.mode = "invalid"

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_log_level(self, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        config = Config()
        assert config.log_level == "DEBUG"

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)

    # ============================================================================================ #
    def test_set_log_level(self, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        config = Config()
        config.log_level = "INFO"
        assert config.log_level == "INFO"

        with pytest.raises(ValueError):
            config.log_level = "invalid"

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_image_directory(self, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        config = Config()
        config.mode = "test"
        assert config.image_directory == "/home/john/projects/bcd/tests/data/images"
        config.mode = "dev"
        assert config.image_directory == "/home/john/projects/bcd/data/image/1_dev"

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)
