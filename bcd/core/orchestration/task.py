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
# Modified   : Saturday October 28th 2023 09:18:59 pm                                              #
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
from bcd.core.base import Application, Entity, Param, Stage
from bcd.utils.date import to_datetime
from bcd.utils.get_class import get_class


# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Task(Entity):
    """Abstract base class for tasks."""

    uid: str
    name: str
    mode: str
    stage_id: int
    stage: str
    application: type[Application]
    application_name: str
    application_module: str
    params: Param
    params_name: str
    params_module: str
    images_processed: int = 0
    image_processing_time: int = 0
    started: datetime = None
    ended: datetime = None
    duration: float = 0
    state: str = "PENDING"
    job_id: str = None

    @classmethod
    def create(
        cls,
        application: type[Application],
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
            name=cls.__name__,
            mode=config().mode,
            stage_id=application.stage,
            stage=Stage(uid=application.stage).name,
            application=application,
            application_name=application.name,
            application_module=application.module,
            params=params,
            params_name=params_name,
            params_module=params_module,
            images_processed=0,
            image_processing_time=0,
            started=None,
            ended=None,
            duration=0,
            state="PENDING",
        )

    def run(self) -> None:
        """Executes the transformer."""
        self.start_task()
        self.application(task_id=self.uid, params=self.params)
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
    def from_df(cls, df: pd.DataFrame) -> Task:
        """Creates a Task object from  a dataframe"""
        application = get_class(
            module_name=df["application_module"].values[0],
            class_name=df["application_name"].values[0],
        )
        if df["params_module"].values[0] is not None:
            params = get_class(
                module_name=df["params_module"].values[0], class_name=df["params_name"].values[0]
            )
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
            application=application,
            application_name=df["application_name"].values[0],
            application_module=df["application_module"].values[0],
            params=params,
            params_name=df["params_name"].values[0],
            params_module=df["params_module"].values[0],
            job_id=df["job_id"].values[0],
            images_processed=df["images_processed"].values[0],
            image_processing_time=df["image_processing_time"].values[0],
            started=started,
            ended=ended,
            duration=df["duration"].values[0],
        )
