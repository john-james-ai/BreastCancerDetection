#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/file.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 29th 2023 01:47:42 am                                                #
# Modified   : Friday December 29th 2023 02:15:17 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""File Database Module"""
import os
from typing import Any

from bcd.dal.base import Database
from bcd.utils.file import IOService


# ------------------------------------------------------------------------------------------------ #
class FileDatabase:
    """File Database for tabular data files."""

    def __init__(self, db_file: str) -> None:
        self._db_file = db_file
        self._registry = IOService.read(db_file)

    def create(self, name: str, data: Any, format: str = "csv", ftype: str = "metadata", stage: str = "exp") -> None:


    @classmethod
    def read_config(cls) -> dict:
        filepath = os.path.abspath(os.getenv("CONFIG_FILEPATH"))
        return IOService.read(filepath=self._filepath)

    @classmethod
    def write_config(cls, config: dict) -> None:
        filepath = os.path.abspath(os.getenv("CONFIG_FILEPATH"))
        IOService.write(filepath=self._filepath, data=config)

    @classmethod
    def get_data_dir(cls) -> str:
        config = cls.read_config()
        return config["data"]["basedir"]

    @classmethod
    def get_raw_metadata_filepath(cls, name) -> str:
        config = cls.read_config()
        return config["data"]["metadata"]["raw"][name]

    @classmethod
    def get_final_metadata_filepath(cls) -> str:
        config = cls.read_config()
        return config["data"]["metadata"]["final"]

    @classmethod
    def get_model_dir(cls) -> str:
        config = cls.read_config()
        return os.path.abspath(config["models"])
