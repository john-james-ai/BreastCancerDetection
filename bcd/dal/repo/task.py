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
# Modified   : Wednesday November 29th 2023 10:58:55 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
from typing import Callable, Union

import pandas as pd
import pymysql
from sqlalchemy.dialects.mssql import DATETIME, INTEGER, VARCHAR

from bcd.config import Config
from bcd.dal.database.base import Database
from bcd.dal.repo.base import Repo
from bcd.preprocess.image.flow.task import Task

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-renamed, arguments-differ, broad-exception-caught
# ------------------------------------------------------------------------------------------------ #
TASK_DTYPES = {
    "uid": VARCHAR(64),
    "mode": VARCHAR(8),
    "stage_id": INTEGER(),
    "stage": VARCHAR(32),
    "method": VARCHAR(64),
    "params": VARCHAR(128),
    "images_processed": INTEGER(),
    "created": DATETIME(),
}


# ------------------------------------------------------------------------------------------------ #
class TaskRepo(Repo):
    """Repository of Tasks"""

    __tablename = "task"

    def __init__(self, database: Database, config: Config = Config) -> None:
        super().__init__(database=database)
        self._mode = config.get_mode()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def mode(self) -> str:
        return self._mode

    def add(self, task: Task) -> None:
        """Adds a task to the repository

        Args:
            task (Task): A Task instance.

        """
        exists = False
        try:
            exists = self.exists(uid=task.uid)
        except pymysql.Error:  # pragma: no cover
            exists = False
        finally:
            if exists:
                msg = f"Task {task.uid} already exists."
                self._logger.exception(msg)
                raise FileExistsError(msg)
            else:
                self.database.insert(
                    data=task.as_df(),
                    tablename=self.__tablename,
                    dtype=TASK_DTYPES,
                    if_exists="append",
                )

    def get(self, uid: str) -> Task:
        """Obtains a task by identifier.

        Args:
            uid (str): Task uid

        Returns:
            Task object.
        """
        query = f"SELECT * FROM {self.__tablename} WHERE uid = :uid;"
        params = {"uid": uid}
        try:
            task = self.database.query(query=query, params=params)
        except Exception as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            if len(task) == 0:
                msg = f"Task {uid} does not exist."
                self._logger.exception(msg)
                raise FileNotFoundError(msg)

            return Task.from_df(df=task)

    def get_by_stage(self, stage_id: int) -> pd.DataFrame:
        """Returns all task for a given stage."""
        tasks = {}
        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode AND stage_id = :stage_id;"
        params = {"mode": self.mode, "stage_id": stage_id}
        task_meta = self.database.query(query=query, params=params)
        if len(task_meta) == 0:
            msg = f"No Tasks exist for Stage{stage_id}."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)
        else:
            for _, meta in task_meta.iterrows():
                task = self.get(uid=meta["uid"])
                tasks[meta["uid"]] = task
        return (task_meta, tasks)

    def get_by_mode(self) -> pd.DataFrame:
        """Returns all task for current mode."""
        tasks = {}
        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        task_meta = self.database.query(query=query, params=params)
        if len(task_meta) == 0:
            msg = f"No Tasks exist for mode {self.mode}."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)
        else:
            for _, meta in task_meta.iterrows():
                task = self.get(uid=meta["uid"])
                tasks[meta["uid"]] = task
        return (task_meta, tasks)

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Returns all task for a given name."""
        tasks = {}
        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode AND name = :name;"
        params = {"mode": self.mode, "name": name}
        task_meta = self.database.query(query=query, params=params)
        if len(task_meta) == 0:
            msg = f"No Tasks exist for Name {name}."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)
        else:
            for _, meta in task_meta.iterrows():
                task = self.get(uid=meta["uid"])
                tasks[meta["uid"]] = task
        return (task_meta, tasks)

    def get_meta(self, condition: Callable = None) -> Union[pd.DataFrame, list]:
        """Returns task metadata
        Args:
            condition (Callable): A lambda expression used to subset the data.
                An example of a condition: condition = lambda df: df['stage_id'] > 0

        """
        query = f"SELECT * FROM {self.__tablename};"
        params = None
        task_meta = self.database.query(query=query, params=params)

        if condition is None:
            return task_meta
        else:
            return task_meta[condition]

    def exists(self, uid: str) -> bool:
        """Evaluates existence of a task by identifier.

        Args:
            id (str): Task UUID

        Returns:
            Boolean indicator of existence.
        """
        query = f"SELECT EXISTS(SELECT 1 FROM {self.__tablename} WHERE uid = :uid);"
        params = {"uid": uid}
        try:
            exists = self.database.exists(query=query, params=params)
        except Exception as e:  # pragma: no cover
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
            task = self.database.query(query=query, params=params)

        except Exception as e:  # pragma: no cover
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
        query = f"DELETE FROM {self.__tablename} WHERE uid = :uid;"
        params = {"uid": uid}
        self.database.delete(query=query, params=params)

    def delete_by_name(self, name: str) -> None:
        """Deletes all tasks with the specified name

        Args:
            name (str): Name of the task.

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode AND name = :name;"
        params = {"mode": self.mode, "name": name}
        self.database.delete(query=query, params=params)

    def delete_by_stage(self, stage_id: int) -> None:
        """Deletes all tasks with the stage_id

        Args:
            stage_id (int): Stage id

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode AND stage_id = :stage_id;"
        params = {"mode": self.mode, "stage_id": stage_id}
        self.database.delete(query=query, params=params)

    def delete_by_mode(self) -> None:
        """Deletes all tasks with the stage_id

        Args:
            stage_id (int): Stage id

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        self.database.delete(query=query, params=params)
