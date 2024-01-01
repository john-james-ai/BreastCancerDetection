#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/noise.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday December 24th 2023 09:45:47 pm                                               #
# Modified   : Sunday December 24th 2023 10:39:13 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import cv2
import numpy as np
from scipy import signal

from bcd import Task


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, arguments-differ
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
#                                      MEAN FILTER                                                 #
# ------------------------------------------------------------------------------------------------ #
class MeanFilter(Task):
    """Mean Filter Denoiser"""

    def __init__(self, kernel: int = 3) -> None:
        super().__init__()
        self._kernel = kernel

    def run(self, image: np.ndarray) -> np.ndarray:
        """Applies the mean filter to an image

        Args:
            image (np.ndarray): Input image
        """

        return cv2.blur(image, (self._kernel, self._kernel))


# ------------------------------------------------------------------------------------------------ #
#                                     GAUSSIAN FILTER                                              #
# ------------------------------------------------------------------------------------------------ #
class GaussianFilter(Task):
    """Gaussian Mean Filter Denoiser"""

    def __init__(self, kernel: int = 3) -> None:
        super().__init__()
        self._kernel = kernel

    def run(self, image: np.ndarray) -> np.ndarray:
        """Applies the Gaussian mean filter to an image

        Args:
            image (np.ndarray): Input image
        """
        return cv2.GaussianBlur(image, (self._kernel, self._kernel))


# ------------------------------------------------------------------------------------------------ #
#                                      MEDIAN FILTER                                               #
# ------------------------------------------------------------------------------------------------ #
class MedianFilter(Task):
    """Median Filter Denoiser"""

    def __init__(self, kernel: int = 3) -> None:
        super().__init__()
        self._kernel = kernel

    def run(self, image: np.ndarray) -> np.ndarray:
        """Applies the median filter to an image

        Args:
            image (np.ndarray): Input image
        """
        return cv2.medianBlur(image, self._kernel)


# ------------------------------------------------------------------------------------------------ #
#                                    BILATERAL FILTER                                              #
# ------------------------------------------------------------------------------------------------ #
class BilateralFilter(Task):
    """Bilateral Filter Denoiser"""

    def __init__(
        self, d: int - 1, sigma_range: float = 25, sigma_domain: float = 25
    ) -> None:
        super().__init__()
        self._d = d
        self._sigma_range = sigma_range
        self._sigma_domain = sigma_domain

    def run(self, image: np.ndarray) -> np.ndarray:
        """Applies the bilateral filter to an image

        Args:
            image (np.ndarray): Input image
        """
        return cv2.bilateralFilter(
            image,
            d=self._d,
            sigmaColor=self._sigma_range,
            sigmaSpace=self._sigma_domain,
        )


# ------------------------------------------------------------------------------------------------ #
#                                    NL MEANS FILTER                                               #
# ------------------------------------------------------------------------------------------------ #
class NLMeansFilter(Task):
    """Non-Local Means Filter Denoiser"""

    def __init__(self, kernel: int = 7, search_window: int = 21, h: int = 10) -> None:
        super().__init__()
        self._kernel = kernel
        self._search_window = search_window
        self._h = h

    def run(self, image: np.ndarray) -> np.ndarray:
        """Applies the bilateral filter to an image

        Args:
            image (np.ndarray): Input image
        """
        return cv2.fastNlMeansDenoising(
            image,
            templateWindowSize=self._kernel,
            searchWindowSize=self._search_window,
            h=self._h,
        )


# ------------------------------------------------------------------------------------------------ #
#                                 BUTTERWORTH FILTER                                               #
# ------------------------------------------------------------------------------------------------ #
class ButterworthFilter(Task):
    """Non-Local Means Filter Denoiser"""

    def __init__(
        self,
        cutoff_freq: int,
        sampling_freq: float = 44100,
        order: int = 10,
        btype: str = "lowpass",
        analog: bool = False,
    ) -> None:
        super().__init__()
        self._cutoff_freq = cutoff_freq
        self._sampling_freq = sampling_freq
        self._order = order
        self._btype = btype
        self._analog = analog

    def run(self, image: np.ndarray) -> np.ndarray:
        """Applies the bilateral filter to an image

        Args:
            image (np.ndarray): Input image
        """
        b, a = signal.butter(
            N=self._order,
            Wn=self._cutoff_freq,
            fs=self._sampling_freq,
            btype=self._btype,
            analog=self._analog,
        )
        return signal.filtfilt(b, a, image)
