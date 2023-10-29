#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/repo/task.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 07:41:24 pm                                              #
# Modified   : Saturday October 28th 2023 10:27:00 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
from typing import Callable

import pandas as pd
import pymysql
from sqlalchemy.dialects.mssql import DATETIME, FLOAT, INTEGER, VARCHAR

from bcd.config import Config
from bcd.core.orchestration.task import Task
from bcd.dal.database.base import Database
from bcd.dal.repo.base import Repo

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-renamed
# ------------------------------------------------------------------------------------------------ #
TASK_DTYPES = {
    "uid": VARCHAR(64),
    "name": VARCHAR(64),
    "mode": VARCHAR(8),
    "stage_uid": INTEGER(),
    "stage": VARCHAR(length=64),
    "started": DATETIME(),
    "ended": DATETIME(),
    "duration": FLOAT(),
    "images_processed": INTEGER(),
    "image_processing_time": FLOAT(),
    "params": VARCHAR(128),
    "state": VARCHAR(16),
}


# ------------------------------------------------------------------------------------------------ #
class TaskRepo(Repo):
    """Repository of Tasks"""

    __tablename = "task"

    def __init__(self, database: Database, config: Config) -> None:
        super().__init__()
        self._database = database
        self._config = config()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def mode(self) -> str:
        return self._config.mode

    def add(self, task: Task) -> None:
        """Adds a task to the repository

        Args:
            task (Task): A Task instance.

        """
        exists = False
        try:
            exists = self.exists(uid=task.id)
        except pymysql.Error:  # pragma: no cover
            exists = False
        finally:
            if exists:
                msg = f"Task {task.id} already exists."
                self._logger.exception(msg)
                raise FileExistsError(msg)
            else:
                self._database.insert(
                    data=task.as_df(),
                    tablename=self.__tablename,
                    dtype=TASK_DTYPES,
                    if_exists="append",
                )

    def get(self, uid: str) -> Task:
        """Obtains a task by identifier.

        Args:
            id (str): Task id

        Returns:
            Task object.
        """
        query = f"SELECT * FROM {self.__tablename} WHERE id = :id;"
        params = {"uid": id}
        try:
            task = self._database.query(query=query, params=params)
        except Exception as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            if len(task) == 0:
                msg = f"Task {uid} does not exist."
                self._logger.exception(msg)
                raise FileNotFoundError(msg)

            return Task.from_df(df=task)

    def get_by_stage(self, stage_uid: int) -> pd.DataFrame:
        """Returns all task for a given stage."""
        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode AND stage_uid = :stage_uid;"
        params = {"mode": self.mode, "stage_uid": stage_uid}
        tasks = self._database.query(query=query, params=params)
        if len(tasks) == 0:
            msg = f"No Tasks exist for Stage{stage_uid}."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)
        return tasks

    def get_by_mode(self) -> pd.DataFrame:
        """Returns all task for current mode."""
        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        tasks = self._database.query(query=query, params=params)
        if len(tasks) == 0:  # pragma: no cover
            msg = f"No Tasks exist for the {self.mode} mode."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)
        return tasks

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Returns all task for a given name."""
        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode AND name = :name;"
        params = {"mode": self.mode, "name": name}
        tasks = self._database.query(query=query, params=params)
        if len(tasks) == 0:
            msg = f"No Tasks exist with name {name}."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)
        return tasks

    def exists(self, uid: str) -> bool:
        """Evaluates existence of a task by identifier.

        Args:
            id (str): Task UUID

        Returns:
            Boolean indicator of existence.
        """
        query = f"SELECT EXISTS(SELECT 1 FROM {self.__tablename} WHERE id = :id);"
        params = {"uid": id}
        try:
            exists = self._database.exists(query=query, params=params)
        except pymysql.Error as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            return exists

    def count(self, condition: Callable = None) -> int:
        """Counts tasks matching the condition

        Args:
            condition (Callable): A lambda expression used to subset the data.

        Returns:
            Integer count of tasks matching condition.
        """
        query = f"SELECT * FROM {self.__tablename};"
        params = None
        try:
            task = self._database.query(query=query, params=params)

        except pymysql.Error as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            if condition is not None:
                task = task[condition]
            return len(task)

    def delete(self, uid: str) -> None:
        """Removes a task given the uid.

        Args:
            uid (str): Task UUID
        """
        query = f"DELETE FROM {self.__tablename} WHERE id = :id;"
        params = {"uid": id}
        self._database.delete(query=query, params=params)

    def delete_by_name(self, name: str) -> None:
        """Deletes all tasks with the specified name

        Args:
            name (str): Name of the task.

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode AND name = :name;"
        params = {"mode": self.mode, "name": name}
        self._database.delete(query=query, params=params)

    def delete_by_stage(self, stage_uid: int) -> None:
        """Deletes all tasks with the stage_uid

        Args:
            stage_uid (int): Stage id

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode AND stage_uid = :stage_uid;"
        params = {"mode": self.mode, "stage_uid": stage_uid}
        self._database.delete(query=query, params=params)

    def delete_by_mode(self) -> None:
        """Deletes all tasks with the stage_uid

        Args:
            stage_uid (int): Stage id

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        self._database.delete(query=query, params=params)

    def delete_all(self) -> None:
        """Deletes all tasks"""
        query = f"DELETE FROM {self.__tablename};"
        params = None
        self._database.delete(query=query, params=params)

    def save(self) -> None:
        """Commits changes to the database, and saves cache to file."""
        self._database.commit()
