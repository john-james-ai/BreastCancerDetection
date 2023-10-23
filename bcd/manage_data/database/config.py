#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/manage_data/database/config.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday August 27th 2023 01:14:57 am                                                 #
# Modified   : Monday October 23rd 2023 03:01:51 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from dotenv import load_dotenv

# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #


class DatabaseConfig:
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
    def autocommit(self) -> bool:
        return os.getenv("MYSQL_AUTOCOMMIT")

    @property
    def timeout(self) -> bool:
        return os.getenv("MYSQL_TIMEOUT")
