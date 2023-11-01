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
# Modified   : Wednesday November 1st 2023 04:31:52 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod

import numpy as np

from bcd.core.base import Param


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
