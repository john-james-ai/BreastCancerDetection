#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/method/basemethod.py                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 31st 2023 05:30:31 am                                               #
# Modified   : Wednesday November 1st 2023 08:41:26 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from bcd import DataClass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Param(DataClass):
    """Abstract base class containing parameters for an instance of an method."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def module(self) -> str:
        return inspect.getmodule(self).__name__

    def as_string(self) -> str:
        d = self.as_dict()
        return json.dumps(d)

    @classmethod
    def from_string(cls, params: str) -> Param:
        d = json.loads(params)
        return cls(**d)


# ------------------------------------------------------------------------------------------------ #
class Method(ABC):
    """Abstract base class for Preprocessing Methods.

    Methods operate on the pixel data for a single image.

    Override the following class variables in the subclasses for identification
    of the class within an composing Task object.

    """

    name = None  # To be overridden by subclasses.
    stage = None  # To be overridden by subclasses.
    step = None  # To be overridden by subclasses.

    @classmethod
    @abstractmethod
    def execute(cls, image: np.ndarray, params: Param) -> np.ndarray:
        """Executes the method"""
