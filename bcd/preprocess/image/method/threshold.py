#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/method/threshold.py                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 30th 2023 04:50:27 pm                                                #
# Modified   : Monday November 13th 2023 01:58:10 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Threshold Methods Module"""
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-name-in-module, arguments-differ
# ------------------------------------------------------------------------------------------------ #
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_niblack,
    threshold_otsu,
    threshold_sauvola,
    threshold_triangle,
)

from bcd import DataClass, Stage
from bcd.preprocess.image.method.base import Method


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, unused-argument
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ThresholdParams(DataClass):
    """Parameter object for Threshold methods."""

    nbins: int = 256
    return_all: bool = False
    max_num_iter: int = 10000
    window_size: int = 15
    k: float = 0.2
    classes: int = 2
    tolerance: float = None


# ------------------------------------------------------------------------------------------------ #
class Threshold(Method):
    """Abstract base class for Threshold methods."""

    name = __qualname__
    stage = Stage(uid=2)

    @classmethod
    @abstractmethod
    def execute(cls, image: np.ndarray, params: ThresholdParams) -> np.ndarray:
        """Executes the method"""


# ------------------------------------------------------------------------------------------------ #
class ThresholdISOData(Threshold):
    """Performs ISO Data Thresholding"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: ThresholdParams) -> np.ndarray:
        return threshold_isodata(
            image.pixel_data, nbins=params.nbins, return_all=params.return_all
        )


# ------------------------------------------------------------------------------------------------ #
class ThresholdLi(Threshold):
    """Performs Li Thresholding"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: ThresholdParams) -> np.ndarray:
        return threshold_li(image.pixel_data, tolerance=params.tolerance)


# ------------------------------------------------------------------------------------------------ #
class ThresholdMean(Threshold):
    """Performs Mean Thresholding"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: ThresholdParams) -> np.ndarray:
        return threshold_mean(image.pixel_data)


# ------------------------------------------------------------------------------------------------ #
class ThresholdNiblack(Threshold):
    """Performs Niblack Thresholding"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: ThresholdParams) -> np.ndarray:
        return threshold_niblack(
            image.pixel_data, window_size=params.window_size, k=params.k
        )


# ------------------------------------------------------------------------------------------------ #
class ThresholdOtsu(Threshold):
    """Performs Otsu's Thresholding"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: ThresholdParams) -> np.ndarray:
        return threshold_otsu(image.pixel_data, nbins=params.nbins)


# ------------------------------------------------------------------------------------------------ #
class ThresholdSauvola(Threshold):
    """Performs Sauvola Thresholding"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: ThresholdParams) -> np.ndarray:
        return threshold_sauvola(
            image.pixel_data, window_size=params.window_size, k=params.k
        )


# ------------------------------------------------------------------------------------------------ #
class ThresholdTriangle(Threshold):
    """Performs Triangle Thresholding"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: ThresholdParams) -> np.ndarray:
        return threshold_triangle(image.pixel_data, nbins=params.nbins)
