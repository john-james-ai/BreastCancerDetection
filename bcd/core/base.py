#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /bcd/core/base.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 31st 2023 07:36:47 pm                                               #
# Modified   : Friday October 27th 2023 06:05:31 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Package Base Module"""
from __future__ import annotations
import os
import inspect
import string
from datetime import datetime
from abc import ABC, abstractmethod, abstractclassmethod
import logging
from dataclasses import dataclass
from typing import Callable
import json

import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------------------------ #
IMMUTABLE_TYPES: tuple = (
    str,
    int,
    float,
    bool,
    np.int16,
    np.int32,
    np.int64,
    np.int8,
    np.uint8,
    np.uint16,
    np.float16,
    np.float32,
    np.float64,
    np.float128,
    np.bool_,
    datetime,
)
SEQUENCE_TYPES: tuple = (
    list,
    tuple,
)
# ------------------------------------------------------------------------------------------------ #
NUMERIC_TYPES = [
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    np.int16,
    np.int32,
    np.int64,
    np.int8,
    np.float16,
    np.float32,
    np.float64,
    np.float128,
    np.number,
    int,
    float,
    complex,
]
# ------------------------------------------------------------------------------------------------ #
STAGES = {
    0: "Converted",
    1: "Artifact Removal",
    2: "Pectoral Removal",
    3: "Denoise",
    4: "Fuzziness",
    5: "Enhance",
    6: "ROI Segmentation",
    7: "Augment",
    8: "Reshape"
}


# ------------------------------------------------------------------------------------------------ #
@dataclass()
class Stage:
    """Encapsulates a stage in the preprocessing and modeling phases."""
    uid: int
    name: str = None

    def __post_init__(self) -> None:
        try:
            self.name = STAGES[self.uid]
        except KeyError:
            msg = f"{self.uid} is an invalid stage id."
            logging.exception(msg)
            raise


# ------------------------------------------------------------------------------------------------ #
NON_NUMERIC_TYPES = ["category", "object"]


# ------------------------------------------------------------------------------------------------ #
class Application(ABC):
    """base class for all application objects."""

    def __init__(self, paramset: ParamSet, task_id: str) -> None:
        self._name = self.__class__.__name__
        self._module = inspect.getmodule(self).__name__

    @property
    def name(self) -> str:
        return self._name

    @property
    def module(self) -> str:
        return self._module

    @property
    @abstractclassmethod
    def stage(cls) -> Stage:
        """Returns the Stage object corresponding to the application"""

# ------------------------------------------------------------------------------------------------ #
@dataclass(eq=False)
class DataClass(ABC):
    """A dataclass with extensions for equality checks, string representation, and formatting."""
    def __eq__(self, other: DataClass) -> bool:
        for key, value in self.__dict__.items():
            if type(value) in IMMUTABLE_TYPES:
                if value != other.__dict__[key]:
                    return False
            elif isinstance(value, np.ndarray):
                if not np.array_equal(value, other.__dict__[key]):
                    return False
            elif isinstance(value, (pd.DataFrame, pd.Series)):
                if not self.__dict__[key].equals(other.__dict__[key]):
                    return False

        return True

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
        """Returns a dictionary representation of the the Config object."""
        return {
            k: self._export_config(v) for k, v in self.__dict__.items() if not k.startswith("_")
        }

    @classmethod
    def _export_config(cls, v):  # pragma: no cover
        """Returns v with Configs converted to dicts, recursively."""
        if isinstance(v, IMMUTABLE_TYPES):
            return v
        elif isinstance(v, SEQUENCE_TYPES):
            return type(v)(map(cls._export_config, v))
        elif isinstance(v, datetime):
            return v
        elif isinstance(v, dict):
            return v
        elif hasattr(v, "as_dict"):
            return v.as_dict()

    def as_df(self) -> pd.DataFrame:
        """Returns the project in DataFrame format"""
        d = self.as_dict()
        return pd.DataFrame(data=d, index=[0])


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Params(DataClass):
    """Abstract base class for preprocessor parameters."""



# ------------------------------------------------------------------------------------------------ #
@dataclass
class Entity(DataClass):
    """Abstract base class for project entities, such as Image, Task and Job."""
    uid: str
    name: str


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    """Provides base class for all repositories classes.

    Args:
        name (str): Repository name. This will be the name of the underlying database table.
        database(Database): Database containing data to access.
    """

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """Adds an entity to the repository

        Args:
            entity (Entity): An entity object
        """

    @abstractmethod
    def get(self, uid: str) -> Entity:
        """Gets an an entity by its identifier.

        Args:
            id (str): Entity identifier.
        """

    @abstractmethod
    def exists(self, uid: str) -> bool:
        """Evaluates existence of an entity by identifier.

        Args:
            id (str): Entity UUID

        Returns:
            Boolean indicator of existence.
        """

    @abstractmethod
    def count(self, condition: Callable = None) -> int:  # noqa
        """Counts the entities matching the criteria. Counts all entities if id is None.

        Args:
            condition (Callable): A lambda expression used to subset the data.

        Returns:
            Integer number of rows matching criteria
        """

    @abstractmethod
    def delete(self, uid: str, *args, **kwargs) -> None:
        """Deletes the entity or entities matching condition.

        Args:
            id (str): Entity identifier.

        """

    @abstractmethod
    def save(self) -> None:
        """Saves changes to database."""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ParamSet(DataClass):
    """Abstract base class containing parameters for an instance of an application."""
    name: str = __qualname__
    module: str = None

    def __post_init__(self) -> None:
        self.module = inspect.getmodule(self).__name__

    def as_string(self) -> str:
        d = self.as_dict()
        return json.dumps(d)

    @classmethod
    def from_string(cls, params: str) -> Params:
        d = json.loads(params)
        return cls(**d)