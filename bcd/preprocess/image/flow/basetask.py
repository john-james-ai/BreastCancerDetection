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
# Modified   : Wednesday November 1st 2023 05:25:44 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the Interface for Task classes."""
from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import pandas as pd

from bcd.config import Config
from bcd.core.base import DataClass, Param
from bcd.dal.io.image import ImageReader
from bcd.preprocess.image.flow.state import State
from bcd.preprocess.image.method.basemethod import Method
from bcd.utils.date import to_datetime


# ------------------------------------------------------------------------------------------------ #
#                                 STATE DECORATOR                                                  #
# ------------------------------------------------------------------------------------------------ #
def timer(func):
    """Wrapper for Task subclass run methods that automatically calls the start and stop methods."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.start()
        result = func(self, *args, **kwargs)
        self.end()
        return result

    return wrapper


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Tasks apply a single Method object and its parameters to a series of images."""

    def __init__(self, state: State = State, config: Config = Config) -> None:
        self._uid = str(uuid4())
        self._state = state()
        self._config = config
        self._image_reader = None
        self._method = None
        self._params = None
        self._job_id = None

    @property
    def image_reader(self) -> ImageReader:
        return self._image_reader

    @property
    def method(self) -> Method:
        return self._method

    @property
    def params(self) -> Param:
        return self._params

    @property
    def job_id(self) -> str:
        return self._job_id

    @job_id.setter
    def job_id(self, job_id: str) -> None:
        self._job_id = job_id

    @property
    @abstractmethod
    def images_processed(self) -> int:
        """Returns the number of images processed."""

    @abstractmethod
    def run(self) -> None:
        """Runs the TAsk"""

    def add_image_reader(self, image_reader: ImageReader) -> None:
        """Adds an ImageReader to the Task"""
        self._image_reader = image_reader

    def add_method(self, method: Method) -> None:
        """Adds a Method class to the Task"""
        self._method = method

    def add_params(self, params: Param) -> None:
        """Adds a Param object for the method to the Task"""
        self._params = params

    def start(self) -> None:
        self._state.start()

    def stop(self) -> None:
        self._state.end()

    def as_dto(self) -> TaskDTO:
        return TaskDTO(
            uid=self._uid,
            name=self.__class__.__name__,
            mode=self._config.get_mode(),
            stage_id=self._method.stage.uid,
            stage=self._method.stage.name,
            method=self._method.__name__,
            params=self._params.as_string(),
            images_processed=self.images_processed,
            image_processing_time=self.images_processed / self._state.duration,
            started=self._state.started,
            stopped=self._state.stopped,
            duration=self._state.duration,
            status=self._state.status,
            job_id=self._job_id,
        )


# ------------------------------------------------------------------------------------------------ #
#                                                                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskDTO(DataClass):
    """Abstract base class for tasks."""

    uid: str
    name: str
    mode: str
    stage_id: int
    stage: str
    method: str
    params: str
    images_processed: int = 0
    image_processing_time: int = 0
    started: datetime = None
    stopped: datetime = None
    duration: float = 0
    status: str = "PENDING"
    job_id: str = None

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> TaskDTO:
        """Creates a Task object from  a dataframe"""

        started = to_datetime(df["started"].values[0])
        stopped = to_datetime(df["stopped"].values[0])

        return cls(
            uid=df["uid"].values[0],
            name=df["name"].values[0],
            mode=df["mode"].values[0],
            stage_id=df["stage_id"].values[0],
            stage=df["stage"].values[0],
            method=df["method"].values[0],
            params=df["params"].values[0],
            images_processed=df["images_processed"].values[0],
            image_processing_time=df["image_processing_time"].values[0],
            started=started,
            stopped=stopped,
            duration=df["duration"].values[0],
            status=df["status"].values[0],
            job_id=df["job_id"].values[0],
        )
