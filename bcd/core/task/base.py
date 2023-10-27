#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/task/base.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 11:03:59 pm                                             #
# Modified   : Thursday October 26th 2023 09:25:43 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from uuid import uuid4
from abc import abstractclassmethod
import importlib
from dataclasses import dataclass
from datetime import datetime
import logging

import pandas as pd
from bcd.core.base import Entity, Application, Params
from bcd.config import Config


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Task(Entity):
    """Abstract base class for tasks."""

    uid: str
    name: str
    application: Application
    mode: str
    stage_uid: int
    stage: str
    module: str
    params: str
    images_processed: int = 0
    image_processing_time: int = 0
    started: datetime = None
    ended: datetime = None
    duration: float = 0
    state: str = "PENDING"

    @classmethod
    def create(cls, application: type[Application], params: Params, config: type[Config]) -> Task:
        """Creates a Task object"""
        uid = str(uuid4())
        config = config()

        return cls(
            uid=uid,
            name=cls.__name__,
            application=application.__name__,
            mode=config.mode,
            stage_uid=application.STAGE.uid,
            stage=application.STAGE.name,
            module=application.MODULE,
            params=params,
            images_processed=0,
            image_processing_time=0,
            started=None,
            ended=None,
            duration=0,
            state="PENDING",
        )

    def run(self) -> None:
        """Executes the preprocessor."""
        self.start_task()

        application = self.get_class(
            module_name=self.module,
            class_name=self.application,
        )
        app = application(params=self.params, task_id=self.uid)
        try:
            app.execute()
            self.images_processed = app.images_processed
        except Exception as e:
            self.crash_task()
            logging.exception(e)
            raise
        else:
            self.end_task()

    def start_task(self) -> None:
        self.started = datetime.now()
        self.state = "STARTED"

    def end_task(self) -> None:
        self.ended = datetime.now()
        self.duration = (self.ended - self.started).total_seconds()
        try:
            self.image_processing_time = self.images_processed / self.duration
        except ZeroDivisionError:
            self.image_processing_time = 0
        self.state = "SUCCESS"

    def crash_task(self) -> None:
        self.ended = datetime.now()
        self.duration = (self.ended - self.started).total_seconds()
        try:
            self.image_processing_time = self.images_processed / self.duration
        except ZeroDivisionError:
            self.image_processing_time = 0
        self.state = "EXCEPTION"

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Task:
        """Creates a Task object from  a dataframe"""
        params = cls.get_params(df["params"].values[0])

        started = (
            df["started"].values[0].astype(datetime)
            if df["started"].values[0] is not None
            else None
        )
        ended = (
            df["ended"].values[0].astype(datetime) if df["ended"].values[0] is not None else None
        )

        return cls(
            uid=df["uid"].values[0],
            name=df["name"].values[0],
            application=df["application"].values[0],
            mode=df["mode"].values[0],
            stage_uid=df["stage_uid"].values[0],
            stage=df["stage"].values[0],
            module=df["module"].values[0],
            params=params,
            images_processed=df["images_processed"].values[0],
            image_processing_time=df["image_processing_time"].values[0],
            started=started,
            ended=ended,
            duration=df["duration"].values[0],
            state=df["state"].values[0],
        )

    @classmethod
    def get_class(cls, module_name: str, class_name: str) -> type[Application]:
        """Converts a string to a class instance."""
        try:
            module = importlib.import_module(module_name)
            try:
                class_ = getattr(module, class_name)
            except AttributeError:
                logging.exception("Class does not exist")
                raise
        except ImportError:
            logging.exception("Module does not exist")
            raise
        return class_ or None

    @abstractclassmethod
    def get_params(cls, params: str) -> Params:
        """Returns a parameter object for the underlying task."""

    def as_df(self) -> pd.DataFrame:
        d = {
            "uid": self.uid,
            "name": self.name,
            "application": self.application,
            "mode": self.mode,
            "stage_uid": self.stage_uid,
            "stage": self.stage,
            "module": self.module,
            "params": self.params.as_string(),
            "images_processed": self.images_processed,
            "image_processing_time": self.image_processing_time,
            "started": self.started,
            "ended": self.ended,
            "duration": self.duration,
            "state": self.state,
        }
        return pd.DataFrame(data=d, index=[0])
