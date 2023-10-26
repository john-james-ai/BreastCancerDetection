#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/task/entity.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 11:03:59 pm                                             #
# Modified   : Thursday October 26th 2023 04:09:49 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import abstractmethod, abstractclassmethod
from uuid import uuid4
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

    id: str
    name: str
    application: Application
    mode: str
    stage_id: int
    stage: str
    module: str
    params: str
    images_processed: int = 0
    image_processing_time: int = 0
    started: datetime = None
    ended: datetime = None
    duration: float = 0
    state: str = "PENDING"

    @abstractmethod
    def run(self) -> None:
        """Runs the task."""

    @classmethod
    def create(cls, application: type[Application], params: Params, config: Config) -> Task:
        """Creates a Task object"""
        id = str(uuid4())
        config = config()

        return cls(
            id=id,
            name=application.__name__,
            application=application,
            mode=config.get_mode(),
            stage_id=application.STAGE.id,
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

    @abstractclassmethod
    def from_df(cls, df: pd.DataFrame) -> Task:
        """Creates a Task object from  a dataframe"""

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
    def _get_class(cls, module_name: str, class_name: str) -> type[Application]:
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

    def as_df(self) -> pd.DataFrame:
        d = {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "stage_id": self.stage_id,
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
