#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/repo/uow.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 26th 2023 01:10:10 am                                              #
# Modified   : Sunday October 29th 2023 02:23:08 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Unit of Work Module"""
import logging

from bcd.core.image.factory import ImageFactory
from bcd.dal.database.base import Database
from bcd.dal.io.image import ImageIO
from bcd.dal.repo.image import ImageRepo
from bcd.dal.repo.task import TaskRepo


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
        image_factory: ImageFactory,
        image_repo: type[ImageRepo],
        task_repo: type[TaskRepo],
        io: ImageIO,
        mode: str,
    ) -> None:
        self._database = database
        self._image_factory = image_factory
        self._image_repo = image_repo
        self._task_repo = task_repo
        self._io = io
        self._mode = mode

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def database(self) -> Database:
        return self._database

    @property
    def image_repo(self) -> ImageRepo:
        return self._image_repo(
            database=self._database,
            image_factory=self._image_factory,
            io=self._io,
            mode=self._mode,
        )

    @property
    def task_repo(self) -> TaskRepo:
        return self._task_repo(database=self._database, mode=self._mode)

    def connect(self) -> None:
        """Connects the database"""
        self._database.connect()

    def begin(self) -> None:
        """Begin a transaction"""
        self._database.begin()

    def save(self) -> None:
        """Saves changes to the underlying sqlite context"""
        self._database.commit()

    def rollback(self) -> None:
        """Returns state of sqlite to the point of the last commit."""
        self._database.rollback()

    def close(self) -> None:
        """Closes the sqlite connection."""
        self._database.close()
