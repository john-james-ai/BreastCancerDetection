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
# Created    : Sunday October 29th 2023 01:47:42 am                                                #
# Modified   : Sunday October 29th 2023 02:22:03 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Configuration Manager Module"""
import logging
import os

from dotenv import load_dotenv

from bcd.dal.io.file import IOService

# ------------------------------------------------------------------------------------------------ #
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
MODES = ["test", "dev", "prod"]
# ------------------------------------------------------------------------------------------------ #
load_dotenv()


# ------------------------------------------------------------------------------------------------ #
class Config:
    """Configuration Manager."""

    def __init__(self) -> None:
        self._config_filepath = os.getenv("CONFIG_FILEPATH")
        self._config = IOService.read(filepath=self._config_filepath)
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def mode(self) -> str:
        return self._config["mode"]

    @mode.setter
    def mode(self, mode) -> None:
        if mode not in MODES:
            msg = f"{mode} is not valid. Valid values are: {MODES}"
            self._logger.exception(msg)
            raise ValueError(msg)
        self._config["mode"] = mode
        IOService.write(filepath=self._config_filepath, data=self._config)

    @property
    def name(self) -> str:
        return os.getenv("MYSQL_DBNAME")

    @property
    def username(self) -> str:
        return os.getenv("MYSQL_USERNAME")

    @property
    def password(self) -> str:
        return os.getenv("MYSQL_PWD")

    @property
    def startup(self) -> str:
        return os.getenv("MYSQL_STARTUP_SCRIPT")

    @property
    def backup_directory(self) -> str:
        return os.getenv("MYSQL_BACKUP_DIRECTORY")

    @property
    def max_attempts(self) -> bool:
        return int(os.getenv("MYSQL_MAX_ATTEMPTS"))

    @property
    def autocommit(self) -> bool:
        autocommit = os.getenv("MYSQL_AUTOCOMMIT")
        return "True" == autocommit

    @property
    def timeout(self) -> bool:
        return int(os.getenv("MYSQL_TIMEOUT"))

    @property
    def log_level(self) -> str:
        return self._config["logging"]["handlers"]["console"]["level"]

    @log_level.setter
    def log_level(self, level: str) -> None:
        if level not in LOG_LEVELS:
            msg = f"{level} is not valid. Valid log levels are: {LOG_LEVELS}"
            self._logger.exception(msg)
            raise ValueError(msg)
        self._config["logging"]["handlers"]["console"]["level"] = level
        IOService.write(filepath=self._config_filepath, data=self._config)
