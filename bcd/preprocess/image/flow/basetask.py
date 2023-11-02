#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/basetask.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 11:03:59 pm                                             #
# Modified   : Wednesday November 1st 2023 08:24:41 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the Interface for Task classes."""
from __future__ import annotations

import inspect
import logging
import string
from abc import ABC, abstractmethod
from uuid import uuid4

import pandas as pd

from bcd.config import Config
from bcd.dal.repo.base import Repo
from bcd.preprocess.image.flow.state import Stage, State
from bcd.preprocess.image.method.basemethod import Method, Param
from bcd.utils.date import to_datetime
from bcd.utils.object import get_class


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Abstract base class for task objects."""

    def __init__(self, state: State = State, config: Config = Config) -> None:
        self._uid = str(uuid4())
        self._name = self.__class__.__name__
        self._mode = config.get_mode()
        self._state = state()
        self._config = config
        self._task_params = None
        self._stage = None
        self._method = None
        self._method_params = None

        self._image_repo = None
        self._job_id = None

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def __eq__(self, other: Task) -> bool:
        return self.uid == other.uid

    def __repr__(self) -> str:  # pragma: no cover tested, but missing in coverage
        s = "{}({})".format(
            self.__class__.__name__,
            ", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )
        return s

    def __str__(self) -> str:
        width = 32
        breadth = width * 2
        s = f"\n\n{self.__class__.__name__.center(breadth, ' ')}"
        d = self.as_dict()
        for k, v in d.items():
            k = string.capwords(
                k.replace(
                    "_",
                    " ",
                )
            )
            s += f"\n{k.rjust(width,' ')} | {v}"
        s += "\n\n"
        return s

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def uow(self) -> str:
        return self._uow

    @uow.setter
    def uow(self, uow: Repo) -> None:
        self._uow = uow

    @property
    def method(self) -> Method:
        return self._method

    @method.setter
    def method(self, method: Method) -> None:
        self._method = method
        self._stage = method.stage

    @property
    def method_params(self) -> Param:
        return self._method_params

    @method_params.setter
    def method_params(self, method_params: Param) -> None:
        self._method_params = method_params

    @property
    def task_params(self) -> Param:
        return self._task_params

    @task_params.setter
    def task_params(self, task_params: Param) -> None:
        self._task_params = task_params

    @property
    def job_id(self) -> str:
        return self._job_id

    @job_id.setter
    def job_id(self, job_id: str) -> None:
        self._job_id = job_id

    @abstractmethod
    def run(self) -> None:
        """Runs the Task"""

    def start(self) -> None:
        """Starts the timer on the state object."""
        self._state.start()

    def count(self) -> None:
        self._state.count()

    def stop(self) -> None:
        """Stops the timer, computes duration, and sets status to Complete"""
        self._state.stop()

    def as_dict(self) -> dict:
        return {
            "uid": self._uid,
            "name": self._name,
            "mode": self._mode,
            "stage_id": self._stage.uid,
            "stage": self._stage.name,
            "method": self._method.__class__.__name__,
            "method_module": inspect.getmodule(self._method).__name__,
            "method_params": self._method_params.__class__.__name__
            if self._method_params is not None
            else None,
            "method_params_module": inspect.getmodule(self._method_params).__name__
            if self._method_params is not None
            else None,
            "method_params_string": self._method_params.as_string()
            if self._method_params is not None
            else None,
            "task_params": self._task_params.__class__.__name__
            if self._task_params is not None
            else None,
            "task_params_module": inspect.getmodule(self._task_params).__name__
            if self._task_params is not None
            else None,
            "task_params_string": self._task_params.as_string()
            if self._task_params is not None
            else None,
            "images_processed": self._state.images_processed,
            "image_processing_time": self._state.image_processing_time,
            "started": self._state.started,
            "stopped": self._state.stopped,
            "duration": self._state.duration,
            "status": self._state.status,
            "job_id": self._job_id,
        }

    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.as_dict(), index=[0])

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Task:
        task = cls()
        # Reconstitute composed objects.
        task._uid = df["uid"].values[0]
        task._name = df["name"].values[0]
        task._mode = df["mode"].values[0]

        task._stage = Stage(uid=df["stage_id"].values[0], name=df["stage"].values[0])
        task._method = get_class(
            class_name=df["method"].values[0], module_name=df["method_module"].values[0]
        )
        task.method_params = None
        if df["method_params"].values[0] is not None:
            method_params = get_class(
                class_name=df["method_params"].values[0],
                module_name=df["method_params_module"].values[0],
            )
            task.method_params = method_params.from_string(df["method_params_string"].values[0])

        task.task_params = None
        if df["task_params"].values[0] is not None:
            task_params = get_class(
                class_name=df["task_params"].values[0],
                module_name=df["task_params_module"].values[0],
            )

            task.task_params = task_params.from_string(df["task_params_string"].values[0])

        task._state = State(
            images_processed=df["images_processed"].values[0],
            image_processing_time=df["image_processing_time"].values[0],
            started=to_datetime(df["started"].values[0]),
            stopped=to_datetime(df["stopped"].values[0]),
            duration=df["duration"].values[0],
            status=df["status"].values[0],
        )
        task._job_id = df["job_id"].values[0]
        return task
