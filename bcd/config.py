#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/config.py                                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 04:01:35 pm                                             #
# Modified   : Thursday October 26th 2023 01:14:03 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Configuration Manager Module"""
import os
import logging
from dotenv import load_dotenv

from bcd.infrastructure.io.file import IOService

# ------------------------------------------------------------------------------------------------ #
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
MODES = ["test", "dev", "prod"]
# ------------------------------------------------------------------------------------------------ #
load_dotenv()


# ------------------------------------------------------------------------------------------------ #
class Config:
    def __init__(self) -> None:
        self._config_filepath = os.getenv("CONFIG_FILEPATH")
        self._config = IOService.read(filepath=self._config_filepath)
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def get_mode(self) -> str:
        return self._config["mode"]

    def set_mode(self, mode) -> None:
        if mode not in MODES:
            msg = f"{mode} is not valid. Valid values are: {MODES}"
            self._logger.exception(msg)
            raise ValueError(msg)
        self._config["mode"] = mode
        IOService.write(filepath=self._config_filepath, data=self._config)

    def get_log_level(self) -> str:
        return self._config["logging"]["handlers"]["console"]["level"]

    def set_log_level(self, level: str) -> None:
        if level not in LOG_LEVELS:
            msg = f"{level} is not valid. Valid log levels are: {LOG_LEVELS}"
            self._logger.exception(msg)
            raise ValueError(msg)
        self._config["logging"]["handlers"]["console"]["level"] = level
        IOService.write(filepath=self._config_filepath, data=self._config)

    def get_image_directory(self) -> str:
        mode = self.get_mode()
        return os.getenv(mode)
