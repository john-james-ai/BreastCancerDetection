#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /bcd/__init__.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 31st 2023 07:35:50 pm                                               #
# Modified   : Tuesday February 13th 2024 05:14:03 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import json
import logging
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=unused-argument
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
NON_NUMERIC_TYPES = ["category", "object"]


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
        width = 50
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
        return {
            k: self._export_config(v)
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

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
