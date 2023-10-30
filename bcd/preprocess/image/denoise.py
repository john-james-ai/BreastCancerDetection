#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/denoise.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 30th 2023 04:50:27 pm                                                #
# Modified   : Monday October 30th 2023 06:33:59 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

import cv2
import numpy as np

from bcd.core.base import Method, Param


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
@dataclass
class FilterParams(Param):
    """Parameter object for Filter methods."""

    kernel: int = 5


# ------------------------------------------------------------------------------------------------ #
class MeanFilter(Method):
    """Performs Mean Filtering"""

    @classmethod
    def execute(cls, image: np.ndarray, kernel: int = 5) -> np.ndarray:
        return cv2.blur(image.pixel_data, (kernel, kernel))


# ------------------------------------------------------------------------------------------------ #
class MedianFilter(Method):
    """Performs Median Filtering"""

    @classmethod
    def execute(cls, image: np.ndarray, kernel: int = 5) -> np.ndarray:
        return cv2.medianBlur(image.pixel_data, kernel)


# ------------------------------------------------------------------------------------------------ #
class GaussianFilter(Method):
    """Performs Gaussian Filtering"""

    def __init__(
        self,
        params: FilterParams,
    ) -> None:
        self._kernel = params.kernel

    def execute(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image.pixel_data, (self._kernel, self._kernel), 0)
