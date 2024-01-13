#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday January 11th 2024 07:49:36 am                                              #
# Modified   : Thursday January 11th 2024 12:33:32 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import json
import string
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from bcd import IMMUTABLE_TYPES, SEQUENCE_TYPES


# ------------------------------------------------------------------------------------------------ #
class Job(ABC):
    """Defines the interface for a job."""

    @abstractmethod
    def add_resources(self, *args, **kwargs) -> None:
        """Adds available computational resources to the Job"""

    @abstractmethod
    def add_inputs(self, *args, **kwargs) -> None:
        """Adds job inputs to the Job"""

    @abstractmethod
    def set_outputs(self, *args, **kwargs) -> None:
        """Sets outputs for the Job."""

    @abstractmethod
    def add_task(self, task: Task) -> None:
        """Adds a Task to the Job"""

    @abstractmethod
    def run(self) -> None:
        """Runs the job"""

    @abstractmethod
    def validate(self) -> None:
        """Validates the Job object."""


# ------------------------------------------------------------------------------------------------ #
class JobBuilder(ABC):
    """Defines the interface for a job builder"""

    @property
    @abstractmethod
    def job(self) -> Job:
        """Returns the complete Job object."""

    @abstractmethod
    def set_inputs(self, *args, **kwargs) -> None:
        """Builds the inputs for the job."""

    @abstractmethod
    def set_outputs(self, *args, **kwargs) -> None:
        """Specifies how the job will produce its output"""

    @abstractmethod
    def add_task(self, task: Task) -> None:
        """Adds a task to the Job"""


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Encapsulates the interface for tasks that perform."""

    @abstractmethod
    def run(self, image: np.ndarray) -> Any:
        """Runs the task."""

    def __repr__(self) -> str:  # pragma: no cover tested, but missing in coverage
        s = "{}({})".format(
            self.__class__.__name__,
            ", ".join(
                "{}={!r}".format(k, v)
                for k, v in self.__dict__.items()
                if type(v) in IMMUTABLE_TYPES
            ),
        )
        return s

    def __str__(self) -> str:
        width = 32
        breadth = width * 2
        s = f"\n\n{self.__class__.__name__.center(breadth, ' ')}"
        d = self.as_dict()
        for k, v in d.items():
            if type(v) in IMMUTABLE_TYPES:
                k = string.capwords(
                    k.replace(
                        "_",
                        " ",
                    )
                )
                s += f"\n{k.rjust(width,' ')} | {v}"
        s += "\n\n"
        return s

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the the FileManager object."""
        return {k: self._export_config(v) for k, v in self.__dict__.items()}

    @classmethod
    def _export_config(cls, v):  # pragma: no cover
        """Returns v with FileManagers converted to dicts, recursively."""
        if isinstance(v, IMMUTABLE_TYPES):
            return v
        elif isinstance(v, SEQUENCE_TYPES):
            return type(v)(map(cls._export_config, v))
        elif isinstance(v, datetime):
            return v
        elif isinstance(v, dict):
            return json.dumps(v)
        elif hasattr(v, "as_dict"):
            return v.as_dict()
        elif isinstance(v, Callable):
            return v.__name__
        elif isinstance(v, object):
            return v.__class__.__name__

    def as_df(self) -> pd.DataFrame:
        """Returns the project in DataFrame format"""
        d = self.as_dict()
        return pd.DataFrame(data=d, index=[0])
