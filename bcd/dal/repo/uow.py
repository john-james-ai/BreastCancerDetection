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
# Modified   : Monday November 13th 2023 03:13:11 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Unit of Work Module"""
import logging

from bcd.dal.database.base import Database
from bcd.dal.repo.evaluate import EvalRepo
from bcd.dal.repo.image import ImageRepo
from bcd.dal.repo.task import TaskRepo
from bcd.image import ImageFactory


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
        image_repo: ImageRepo,
        eval_repo: EvalRepo,
        task_repo: TaskRepo,
        image_factory: type[ImageFactory] = ImageFactory,
    ) -> None:
        self._database = database
        self._image_factory = image_factory
        self._image_repo = image_repo
        self._eval_repo = eval_repo
        self._task_repo = task_repo

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def database(self) -> Database:
        return self._database

    @property
    def image_repo(self) -> ImageRepo:
        return self._image_repo(
            database=self._database,
            image_factory=self._image_factory,
        )

    @property
    def task_repo(self) -> TaskRepo:
        return self._task_repo(database=self._database)

    @property
    def eval_repo(self) -> EvalRepo:
        return self._eval_repo(database=self._database)

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

    def reset(self) -> None:
        """Resets all repositories for the current mode."""
        msg = "This will restore all repositories in the current mode to their initial state. Proceed? [Y/N]"
        proceed = input(msg)
        if "y" in proceed.lower():
            self.image_repo.delete_by_mode()
            self.eval_repo.delete_by_mode()
            self.task_repo.delete_by_mode()
