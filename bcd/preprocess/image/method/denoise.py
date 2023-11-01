#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/method/denoise.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 30th 2023 04:50:27 pm                                                #
# Modified   : Wednesday November 1st 2023 09:06:03 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from bcd.preprocess.image.flow.state import Stage
from bcd.preprocess.image.method.basemethod import Method, Param


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
@dataclass
class FilterParams(Param):
    """Parameter object for Filter methods."""

    kernel: int = 5


# ------------------------------------------------------------------------------------------------ #
class Filter(Method):
    """Abstract base class for Filters."""

    name = __qualname__
    stage = Stage(uid=1)
    step = "Denoise"

    @classmethod
    @abstractmethod
    def execute(cls, image: np.ndarray, params: FilterParams) -> np.ndarray:
        """Executes the method"""


# ------------------------------------------------------------------------------------------------ #
class MeanFilter(Filter):
    """Performs Mean Filtering"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: FilterParams) -> np.ndarray:
        return cv2.blur(image.pixel_data, (params.kernel, params.kernel))


# ------------------------------------------------------------------------------------------------ #
class MedianFilter(Filter):
    """Performs Median Filtering"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: FilterParams) -> np.ndarray:
        return cv2.medianBlur(image.pixel_data, params.kernel)


# ------------------------------------------------------------------------------------------------ #
class GaussianFilter(Filter):
    """Performs Gaussian Filtering"""

    name = __qualname__

    @classmethod
    def execute(self, image: np.ndarray, params: FilterParams) -> np.ndarray:
        return cv2.GaussianBlur(image.pixel_data, (params.kernel, params.kernel), 0)
