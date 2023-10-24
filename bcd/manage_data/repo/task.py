#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/manage_data/repo/task.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 07:41:24 pm                                              #
# Modified   : Monday October 23rd 2023 11:32:54 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
from typing import Callable

import pandas as pd
import pymysql
from sqlalchemy.dialects.mssql import VARCHAR, DATETIME, INTEGER, FLOAT, TINYINT, JSON

from bcd.manage_data.repo.base import Repo
from bcd.preprocess.base import Task
from bcd.manage_data.database.base import Database

# ------------------------------------------------------------------------------------------------ #
TASK_DTYPES = {
    "id": VARCHAR(length=64),
    "task": VARCHAR(length=64),
    "mode": VARCHAR(length=8),
    "stage_id": TINYINT(),
    "stage": VARCHAR(length=64),
    "started": DATETIME(),
    "ended": DATETIME(),
    "duration": FLOAT(),
    "tasks_processed": INTEGER(),
    "task_processing_time": FLOAT(),
    "success": TINYINT(),
    "params": JSON(),
}


# ------------------------------------------------------------------------------------------------ #
class TaskRepo(Repo):
    __tablename = "task"

    def __init__(self, database: Database) -> None:
        super().__init__()
        self._database = database
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def database(self) -> Database:
        return self._database

    def add(self, task: Task) -> None:
        """Adds a task to the repository

        Args:
            task (Task): A Task instance.

        """
        try:
            exists = self.exists(id=task.id)
        except Exception:
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

    def get(self, condition: Callable = None) -> pd.DataFrame:
        """Returns task runs matching the condition.

        Args:
            condition (Callable): A lambda expression used to subset the data. If None,
            all tasks will be returned.

        Returns:
            DataFrame containing the tasks meeting the condition.
        """

        query = f"SELECT * FROM {self.__tablename};"
        params = None
        task = self._database.query(query=query, params=params)

        if condition is not None:
            task = task[condition]
        return task

    def exists(self, id: str) -> bool:
        """Evaluates existence of a task by identifier.

        Args:
            id (str): Task UUID

        Returns:
            Boolean indicator of existence.
        """
        query = f"SELECT EXISTS(SELECT 1 FROM {self.__tablename} WHERE id = :id);"
        params = {"id": id}
        try:
            exists = self._database.exists(query=query, params=params)
        except pymysql.Error as e:
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

        except pymysql.Error as e:
            self._logger.exception(e)
            raise
        else:
            if condition is not None:
                task = task[condition]
            return len(task)

    def delete(self, condition: Callable) -> None:
        """Removes tasks matching the condition.

        Args:
            condition (Callable): Lambda expression subsetting the data.
        """
        query = f"SELECT * FROM {self.__tablename}"
        params = None
        task = self._database.query(query=query, params=params)
        task = task[condition]
        ids = tuple(task["id"])

        query = f"DELETE FROM {self.__tablename} WHERE id IN {ids}"
        params = None
        self._database.execute(query=query, params=params)
