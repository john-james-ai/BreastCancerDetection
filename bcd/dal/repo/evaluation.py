#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/repo/evaluation.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 07:41:24 pm                                              #
# Modified   : Tuesday October 31st 2023 12:03:29 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
from typing import Callable

import pandas as pd
from sqlalchemy.dialects.mssql import DATETIME, FLOAT, INTEGER, TINYINT, VARCHAR

from bcd.config import Config
from bcd.dal.database.base import Database
from bcd.dal.repo.base import Repo
from bcd.preprocess.image.evaluate import Evaluation

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-renamed, arguments-differ, broad-exception-caught
# ------------------------------------------------------------------------------------------------ #
EVAL_DTYPES = {
    "image_uid": VARCHAR(64),
    "mode": VARCHAR(8),
    "stage_id": INTEGER,
    "stage": VARCHAR(32),
    "step": VARCHAR(64),
    "method": VARCHAR(64),
    "params": VARCHAR(128),
    "mse": FLOAT,
    "psnr": FLOAT,
    "ssim": FLOAT,
    "image_view": VARCHAR(4),
    "abnormality_type": VARCHAR(24),
    "assessment": INTEGER,
    "cancer": TINYINT,
    "evaluated": DATETIME,
}


# ------------------------------------------------------------------------------------------------ #
class EvalRepo(Repo):
    """Repository of Tasks"""

    __tablename = "eval"

    def __init__(self, database: Database, config: Config = Config) -> None:
        super().__init__()
        self._database = database
        self._mode = config.get_mode()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def mode(self) -> str:
        return self._mode

    def add(self, evaluation: Evaluation) -> None:
        """Adds a task to the repository

        Args:
            evaluation (Evaluation): An Evaluation Instance.

        """
        self._database.insert(
            data=evaluation.as_df(),
            tablename=self.__tablename,
            dtype=EVAL_DTYPES,
            if_exists="append",
        )

    def get(self, condition: Callable = None) -> pd.DataFrame:
        """Returns a DataFrame containing evaluation results.

        Args:
            condition (Callable): A lambda expression used to subset the data.
                An example of a condition:
                    condition = lambda df: df['method'] == 'GaussianFilter'

        Return:
            DataFrame containing evaluation results.
        """
        query = f"SELECT * FROM {self.__tablename};"
        params = {}
        try:
            evals = self._database.query(query=query, params=params)
        except Exception as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            # Filter by current mode
            evals = evals.loc[evals["mode"] == self.mode]
            if condition is None:
                return evals
            else:
                return evals[condition]

    def exists(self, condition: Callable = None) -> bool:
        """Evaluates existence of a task by identifier.

        Args:
            condition (Callable): A lambda expression used to subset the data.
                An example of a condition:
                    condition = lambda df: df['method'] == 'GaussianFilter'
        Returns:
            Boolean indicator of existence.
        """
        query = f"SELECT * FROM {self.__tablename};"
        params = {}
        try:
            evals = self._database.query(query=query, params=params)
        except Exception as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            # Filter by current mode
            evals = evals.loc[evals["mode"] == self.mode]
            if condition is None:
                return len(evals) > 0
            else:
                return len(evals[condition]) > 0

    def count(self, condition: Callable = None) -> int:
        """Counts tasks matching the condition

        Args:
            condition (Callable): A lambda expression used to subset the data.
                An example of a condition:
                    condition = lambda df: df['method'] == 'GaussianFilter'

        Returns:
            Integer count of tasks matching condition.
        """
        query = f"SELECT * FROM {self.__tablename};"
        params = None
        try:
            evals = self._database.query(query=query, params=params)

        except Exception as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            # Filter by current mode
            evals = evals.loc[evals["mode"] == self.mode]
            if condition is not None:
                evals = evals[condition]
            return len(evals)

    def delete(self, uid: str) -> None:
        """Removes a task given the uid.

        Args:
            condition (Callable): A lambda expression used to subset the data.
                An example of a condition:
                    condition = lambda df: df['method'] == 'GaussianFilter'
        """
        raise NotImplementedError(
            "The generic delete method is not implemented for this repository."
        )

    def delete_by_method(self, method: str) -> None:
        """Deletes all evaluations for a given method in the current mode.

        Args:
            method (str): Name of a method

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode AND method = :method;"
        params = {"mode": self.mode, "method": method}
        self._database.delete(query=query, params=params)

    def delete_by_stage(self, stage_id: int) -> None:
        """Deletes all evaluations for a given stage_id

        Args:
            stage_id (int): Stage id

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode AND stage_id = :stage_id;"
        params = {"mode": self.mode, "stage_id": stage_id}
        self._database.delete(query=query, params=params)

    def delete_by_step(self, step: str) -> None:
        """Deletes all evaluations for a given step

        Args:
            step (str): Step name

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode AND step = :step;"
        params = {"mode": self.mode, "step": step}
        self._database.delete(query=query, params=params)

    def delete_by_mode(self) -> None:
        """Deletes all evaluations for the current mode.

        Args:
            stage_id (int): Stage id

        """
        query = f"DELETE FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        self._database.delete(query=query, params=params)
