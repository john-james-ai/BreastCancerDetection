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
# Modified   : Saturday November 4th 2023 04:18:35 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Configuration Manager Module"""
import logging
import os

import dotenv

from bcd.dal.io.file import IOService

# ------------------------------------------------------------------------------------------------ #
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
MODES = ["test", "dev", "prod", "exp"]
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
class Config:
    """Configuration Manager."""

    @classmethod
    def read_config(cls) -> dict:
        filepath = os.path.abspath(os.getenv("CONFIG_FILEPATH"))
        return IOService.read(filepath=filepath)

    @classmethod
    def write_config(cls, config: dict) -> None:
        filepath = os.path.abspath(os.getenv("CONFIG_FILEPATH"))
        IOService.write(filepath=filepath, data=config)

    @classmethod
    def get_mode(cls) -> str:
        dotenv_file = dotenv.find_dotenv()
        return dotenv.get_key(dotenv_file, key_to_get="MODE")

    @classmethod
    def set_mode(cls, mode) -> None:
        if mode not in MODES:
            msg = f"{mode} is not valid. Valid values are: {MODES}"
            logging.exception(msg)
            raise ValueError(msg)
        dotenv_file = dotenv.find_dotenv()
        dotenv.load_dotenv(dotenv_file)
        os.environ["MODE"] = mode
        dotenv.set_key(str(dotenv_file), key_to_set="MODE", value_to_set=mode, export=True)

    @classmethod
    def get_name(cls) -> str:
        return os.getenv("MYSQL_DBNAME")

    @classmethod
    def get_username(cls) -> str:
        return os.getenv("MYSQL_USERNAME")

    @classmethod
    def get_password(cls) -> str:
        return os.getenv("MYSQL_PWD")

    @classmethod
    def get_startup(cls) -> str:
        return os.getenv("MYSQL_STARTUP_SCRIPT")

    @classmethod
    def get_backup_directory(cls) -> str:
        return os.getenv("MYSQL_BACKUP_DIRECTORY")

    @classmethod
    def get_max_attempts(cls) -> bool:
        return int(os.getenv("MYSQL_MAX_ATTEMPTS"))

    @classmethod
    def get_autocommit(cls) -> bool:
        autocommit = os.getenv("MYSQL_AUTOCOMMIT")
        return "True" == autocommit

    @classmethod
    def get_timeout(cls) -> bool:
        return int(os.getenv("MYSQL_TIMEOUT"))

    @classmethod
    def get_log_level(cls) -> str:
        config = cls.read_config()
        return config["logging"]["handlers"]["console"]["level"]

    @classmethod
    def set_log_level(cls, level: str) -> None:
        if level not in LOG_LEVELS:
            msg = f"{level} is not valid. Valid log levels are: {LOG_LEVELS}"
            logging.exception(msg)
            raise ValueError(msg)
        config = cls.read_config()
        config["logging"]["handlers"]["console"]["level"] = level
        cls.write_config(config=config)

    @classmethod
    def get_autoconnect(cls) -> bool:
        autoconnect = os.getenv("MYSQL_AUTOCONNECT")
        return "True" == autoconnect

    @classmethod
    def set_autoconnect(cls) -> bool:
        autoconnect = os.getenv("MYSQL_AUTOCONNECT")
        return "True" == autoconnect

    @classmethod
    def get_data_dir(cls) -> str:
        config = cls.read_config()
        mode = cls.get_mode()
        return os.path.abspath(config["data_dir"][mode])

    @classmethod
    def get_dicom_metadata_filepath(cls) -> str:
        config = cls.read_config()
        return os.path.abspath(config["data_dir"]["metadata"])
