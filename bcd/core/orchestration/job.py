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
# Modified   : Friday October 27th 2023 05:26:11 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Job Module: Defining the the abstract base class command object for pipeline jobs."""
from __future__ import annotations
from uuid import uuid4
from abc import ABC, abstractmethod

from bcd.core.orchestration.task import Task
from bcd.core.base import Application, ParamSet
from bcd.config import Config
# ------------------------------------------------------------------------------------------------ #
class Job(ABC):
    """Encapsulates the command for a pipeline job, providing iteration over Task objects.

    Args:
        application (Application): Abstract base class for any type of application, e.g.
            Preprocessor, Model, etc....
        config (Config): Object providing configuration, such as current mode,
            e.g., ['test','dev','prod']
    """

    @abstractmethod
    def __init__(self, application: Application, config: type[Config] = Config) -> None:
        """Instantiates a Job object."""
        self._application = application
        self._paramsets = []
        self._mode = config().mode
        self._tasks = []
        self._task_idx = 0
        self._uid = str(uuid4())

    @property
    def uid(self) -> str:
        """Returns the job uid"""
        return self._uid

    @property
    def mode(self) -> str:
        return self._mode

    def add_paramset(self, paramset: ParamSet) -> None:
        """Adds a parameter set to the job.

        Each parameter set will define an individual Task object.

        Args:
            paramset (ParamSet): A set of parameters to configure an individual
                Task application configuration.
        """
        self._paramsets.append(paramset)

    def __iter__(self) -> Job:
        """Initializes the Task objects."""
        for paramset in self._paramsets:
            task = Task.create(job=self, application=self._application, paramset=paramset)
            self._tasks.append(task)
        self._task_idx = 0
        return self

    @abstractmethod
    def __next__(self) -> Task:
        """Returns the next Task object in the job."""
        task = self._tasks[self._task_idx]
        self._task_idx += 1
        return task
