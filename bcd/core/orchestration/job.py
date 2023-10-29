#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/orchestration/job.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 27th 2023 04:09:45 pm                                                #
# Modified   : Saturday October 28th 2023 01:37:10 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Job Module: Defining the the abstract base class command object for pipeline jobs."""
from __future__ import annotations
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass
from abc import abstractmethod
from typing import List

from bcd.core.base import DataClass
from bcd.core.orchestration.task import Task
from bcd.config import Config
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Job(DataClass):
    """Encapsulates the command for a pipeline job, providing iteration over Task objects."""
    name: str
    uid: str = None
    mode: str = None
    tasks: List[Task] = []
    started: datetime = None
    ended: datetime = None
    duration: float = 0
    state: str = 'PENDING'

    def __post_init__(self) -> None:
        self.uid = str(uuid4())
        self.mode = Config().mode

    def add_task(self, task: Task) -> None:
        """Adds a Task object to the Job"""
        task.job_id = self.uid
        self.tasks.append(task)

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> Job:
        """Creates an instance of the Job class."""

    @abstractmethod
    def __iter__(self) -> Job:
        """Initializes the Task list."""

    @abstractmethod
    def __next__(self) -> Task:
        """Returns the next Task object in the job."""
