#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/basejob.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 11:03:59 pm                                             #
# Modified   : Sunday November 5th 2023 10:10:25 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the Interface for Task classes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from bcd import DataClass
from bcd.config import Config
from bcd.dal.repo.uow import UoW
from bcd.preprocess.image.flow.basetask import Task
from bcd.preprocess.image.flow.stage import State
from bcd.utils.date import to_datetime


# ------------------------------------------------------------------------------------------------ #
#                                          JOB                                                     #
# ------------------------------------------------------------------------------------------------ #
class Job(ABC):
    """Abstract base class for Job objects."""

    def __init__(self, n_jobs: int = 6, state: State = State) -> None:
        self._uid = str(uuid4())
        self._n_jobs = n_jobs
        self._state = state()
        self._name = self.__class__.__name__
        self._mode = Config.get_mode()
        self._tasks_processed = 0
        self._tasks = []

    @property
    def tasks_processed(self) -> int:
        return self._tasks_processed

    @property
    def uow(self) -> str:
        return self._uow

    @uow.setter
    def uow(self, uow: UoW) -> None:
        self._uow = uow

    def add_task(self, task: Task) -> None:
        """Adds a Task object to the Job

        Args:
            task (Task): Task object
        """
        task.job_id = self._uid

    @abstractmethod
    def run(self) -> None:
        """Executes the job by iterating through the tasks"""
        self._state.start()

        with parallel_backend("loky", inner_max_num_threads=self._n_jobs):
            Parallel(n_jobs=self._n_jobs)(delayed(task.run()) for task in self._tasks)

        self._tasks_processed = len(self._tasks)
        self._state.stop()

    def process_task(self, task: Task) -> None:
        """Executes and persists a task object.

        Args:
            task (Task): A Task object.
        """
        task.run()
        self._uow.connect()
        self._uow.task_repo.add(task=task)

    def start(self) -> None:
        """Starts the timer on the state object."""
        self._state.start()

    def stop(self) -> None:
        """Stops the timer, computes duration, and sets status to Complete"""
        self._state.end()

    def as_dto(self) -> JobDTO:
        """Returns the Task data transfer object."""
        return JobDTO(
            uid=self._uid,
            name=self._name,
            mode=self._mode,
            tasks_processed=self._tasks_processed,
            task_processing_time=self._tasks_processed / self._state.duration,
            started=self._state.started,
            stopped=self._state.stopped,
            duration=self._state.duration,
            status=self._state.status,
        )
