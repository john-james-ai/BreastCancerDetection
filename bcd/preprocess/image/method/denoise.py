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
# Modified   : Saturday November 4th 2023 06:59:10 am                                              #
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
    d: int = 0
    sigma_range_factor: int = 1
    sigma_domain_factor: int = 1


# ------------------------------------------------------------------------------------------------ #
class Filter(Method):
    """Abstract base class for Filters."""

    name = __qualname__
    stage = Stage(uid=1)

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
        return cv2.blur(image, (params.kernel, params.kernel))


# ------------------------------------------------------------------------------------------------ #
class MedianFilter(Filter):
    """Performs Median Filtering"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: FilterParams) -> np.ndarray:
        return cv2.medianBlur(image, params.kernel)


# ------------------------------------------------------------------------------------------------ #
class GaussianFilter(Filter):
    """Performs Gaussian Filtering"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: FilterParams) -> np.ndarray:
        return cv2.GaussianBlur(image, (params.kernel, params.kernel), 0)


# ------------------------------------------------------------------------------------------------ #
class BilateralFilter(Filter):
    """Performs Gaussian Filtering"""

    name = __qualname__

    @classmethod
    def execute(cls, image: np.ndarray, params: FilterParams) -> np.ndarray:
        sigma_color = cls._est_sigma_color(image=image, params=params)
        sigma_space = cls._est_sigma_space(image=image, params=params)

        return cv2.bilateralFilter(
            image,
            d=params.d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
            borderType=cv2.BORDER_DEFAULT,
        )

    @classmethod
    def _est_sigma_color(cls, image: np.ndarray, params: FilterParams) -> float:
        """Estimates sigma color (range) as the median of the image gradient."""
        scale = 1
        delta = 0
        ddepth = cv2.CV_32F
        ksize = 3
        gX = cv2.Sobel(
            image,
            ddepth=ddepth,
            dx=1,
            dy=0,
            ksize=ksize,
            scale=scale,
            delta=delta,
            borderType=cv2.BORDER_DEFAULT,
        )
        gY = cv2.Sobel(
            image,
            ddepth=ddepth,
            dx=0,
            dy=1,
            ksize=ksize,
            scale=scale,
            delta=delta,
            borderType=cv2.BORDER_DEFAULT,
        )
        g = np.concatenate((gX, gY))
        g_median = np.median(g)
        theta_range = g_median * params.sigma_range_factor
        return theta_range

    @classmethod
    def _est_sigma_space(cls, image: np.ndarray, params: FilterParams) -> float:
        """Estimates sigma space (domain) to be 2% of the image diagonal."""
        size = np.sqrt(np.square(image.shape[0]) + np.square(image.shape[1]))
        return 0.02 * size * params.sigma_domain_factor
