#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/orchestration/task.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 11:03:59 pm                                             #
# Modified   : Monday October 30th 2023 04:46:59 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the Interface for Task classes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import pandas as pd

from bcd.config import Config
from bcd.core.base import Entity, Method, Param, Stage
from bcd.utils.date import to_datetime
from bcd.utils.get_class import get_class


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Task(Entity):
    """Abstract base class for tasks."""

    uid: str
    name: str
    mode: str
    stage_id: int
    stage: str
    method: type[Method]
    method_name: str
    method_module: str
    params: Param
    params_string: str
    params_name: str
    params_module: str
    images_processed: int = 0
    image_processing_time: int = 0
    started: datetime = None
    ended: datetime = None
    duration: float = 0
    state: str = "PENDING"
    job_id: str = None

    def run(self) -> None:
        """Executes the transformer."""
        self.start_task()
        app = self.method(task_id=self.uid, params=self.params)
        app.execute()
        self.images_processed = app.images_processed
        self.end_task()

    def start_task(self) -> None:
        self.started = datetime.now()
        self.state = "STARTED"

    def end_task(self) -> None:
        self.ended = datetime.now()
        self.duration = (self.ended - self.started).total_seconds()
        self.image_processing_time = self.images_processed / self.duration
        self.state = "SUCCESS"

    def crash_task(self) -> None:
        self.ended = datetime.now()
        self.duration = (self.ended - self.started).total_seconds()
        self.image_processing_time = self.images_processed / self.duration
        self.state = "EXCEPTION"

    @classmethod
    def create(
        cls,
        method: type[Method],
        params: Param,
        config: type[Config] = Config,
    ) -> Task:
        """Creates a Task object"""
        uid = str(uuid4())

        if params is None:
            params_name = None
            params_module = None
        else:
            params_name = params.name
            params_module = params.module

        return cls(
            uid=uid,
            name=method.__name__,
            mode=config.get_mode(),
            stage_id=method.stage_id,
            stage=Stage(uid=method.stage_id).name,
            method=method,
            method_name=method.name,
            method_module=method.module,
            params=params,
            params_string=params.as_string(),
            params_name=params_name,
            params_module=params_module,
            images_processed=0,
            image_processing_time=0,
            started=None,
            ended=None,
            duration=0,
            state="PENDING",
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Task:
        """Creates a Task object from  a dataframe"""
        method = get_class(
            module_name=df["method_module"].values[0],
            class_name=df["method_name"].values[0],
        )
        if df["params_module"].values[0] is not None:
            params = get_class(
                module_name=df["params_module"].values[0], class_name=df["params_name"].values[0]
            )
            params = params.from_string(params=df["params_string"].values[0])
        else:
            params = None

        started = to_datetime(df["started"].values[0])
        ended = to_datetime(df["ended"].values[0])

        return cls(
            uid=df["uid"].values[0],
            name=df["name"].values[0],
            mode=df["mode"].values[0],
            stage_id=df["stage_id"].values[0],
            stage=df["stage"].values[0],
            method=method,
            method_name=df["method_name"].values[0],
            method_module=df["method_module"].values[0],
            params=params,
            params_string=df["params_string"].values[0],
            params_name=df["params_name"].values[0],
            params_module=df["params_module"].values[0],
            images_processed=df["images_processed"].values[0],
            image_processing_time=df["image_processing_time"].values[0],
            started=started,
            ended=ended,
            duration=df["duration"].values[0],
            state=df["state"].values[0],
            job_id=df["job_id"].values[0],
        )

    def as_df(self) -> pd.DataFrame:
        """Returns the Task as a DataFrame object."""
        d = {
            "uid": self.uid,
            "name": self.name,
            "mode": self.mode,
            "stage_id": self.stage_id,
            "stage": self.stage,
            "method_name": self.method_name,
            "method_module": self.method_module,
            "params_string": self.params_string,
            "params_name": self.params_name,
            "params_module": self.params_module,
            "images_processed": self.images_processed,
            "image_processing_time": self.image_processing_time,
            "started": self.started,
            "ended": self.ended,
            "duration": self.duration,
            "state": self.state,
            "job_id": self.job_id,
        }
        return pd.DataFrame(data=d, index=[0])
