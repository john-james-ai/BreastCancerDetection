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
# Modified   : Wednesday December 27th 2023 09:58:29 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Configuration Manager Module"""
import os

from bcd.utils.file import IOService


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
    def get_data_dir(cls, stage: str) -> str:
        config = cls.read_config()
        return os.path.abspath(config["data"][stage])

    @classmethod
    def get_case_file_urls(cls) -> str:
        config = cls.read_config()
        return config["data"]["source_files"]

    @classmethod
    def get_filepath(cls, name) -> str:
        config = cls.read_config()
        return config["data"]["filepaths"][name]

    @classmethod
    def get_model_dir(cls) -> str:
        config = cls.read_config()
        return os.path.abspath(config["models"])
