#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/uow.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 26th 2023 01:10:10 am                                              #
# Modified   : Thursday October 26th 2023 05:50:01 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Unit of Work Module"""
import logging

from bcd.core.base import Repo
from bcd.config import Config
from bcd.infrastructure.io.cache import ImageCache
from bcd.infrastructure.database.base import Database


# ------------------------------------------------------------------------------------------------ #
#                                       UNIT OF WORK CLASS                                         #
# ------------------------------------------------------------------------------------------------ #
class UoW:
    """Unit of Work class encapsulating the repositories used in project objects.

    Args:
        database (Database): A Database instance from the dependency injector container.
        content (Repo): The content repository
        task (Repo): The task repository

    """

    def __init__(
        self,
        database: Database,
        image_repo: Repo,
        task_repo: Repo,
    ) -> None:
        self._database = database
        self._image_repo = image_repo
        self._task_repo = task_repo

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def database(self) -> Database:
        return self._database

    @property
    def image_repo(self) -> Repo:
        return self._image_repo(database=self._database, config=Config, cache=ImageCache)

    @property
    def task_repo(self) -> Repo:
        return self._task_repo(database=self._database, config=Config)

    def connect(self) -> None:
        """Connects the database"""
        self._database.connect()

    def begin(self) -> None:
        """Begin a transaction"""
        self._database.begin()
        self.image_repo.save()

    def save(self) -> None:
        """Saves changes to the underlying sqlite context"""
        self._database.commit()
        self.image_repo.save()

    def rollback(self) -> None:
        """Returns state of sqlite to the point of the last commit."""
        self._database.rollback()
        self.image_repo.reset()

    def close(self) -> None:
        """Closes the sqlite connection."""
        self._database.close()
