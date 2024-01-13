#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/threshold.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday December 24th 2023 10:44:32 pm                                               #
# Modified   : Thursday January 11th 2024 03:27:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, arguments-differ, unused-argument, no-name-in-module, line-too-long,
# pylint: disable=invalid-name
# ------------------------------------------------------------------------------------------------ #
from abc import abstractmethod
from typing import Union

import cv2
import numpy as np
from skimage.filters import threshold_isodata, threshold_li, threshold_yen

from bcd.preprocess.base import Task


# ------------------------------------------------------------------------------------------------ #
#                                         THRESHOLD                                                #
# ------------------------------------------------------------------------------------------------ #
class Threshold(Task):
    """Base class for Thresholding methods.

    Args:
        name (str): Name of the threshold, defined in subclasses
        global_threshold (bool): Whether the threshold is global.
    """

    def __init__(self, name: str, global_threshold: bool) -> None:
        super().__init__()
        self._name = name
        self._global_threshold = global_threshold

    @property
    def is_global(self) -> str:
        return self._global_threshold

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def run(self, image: np.ndarray) -> Union[float, np.array]:
        """Applies the threshold and returns the threshold (if local) and the binary mask.

        Args:
            image (np.ndarray): Image in numpy format.

        Returns:
            threshold (float): The threshold value
            binary_mask (np.ndarray): Binary mask same size and dimensions as image.
        """


# ------------------------------------------------------------------------------------------------ #
class ThresholdManual(Threshold):
    """Performs manual thresholding.

    Args:
        name (str): Name of the threshold, defined in subclasses
        threshold (Union[int,float]): The threshold as a float in (0,1) or integer. If
            a float, the threshold is converted to the maximum pixel value * threshold
        global_threshold (bool): Whether the threshold is global.
    """

    def __init__(
        self,
        name: str = "Manual Threshold",
        threshold: float = 0.1,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name=name, global_threshold=global_threshold)
        self._threshold = threshold

    def run(self, image: np.ndarray) -> Union[float, np.ndarray]:
        # threshold can be a proportion between on and the maximum pixel value (< 1) or
        # a specific pixel values (>=1)
        if self._threshold < 1:
            threshold = int((np.max(image) - np.min(image)) * self._threshold)
        else:
            threshold = self._threshold

        threshold, image = cv2.threshold(
            image, thresh=threshold, maxval=np.max(image), type=cv2.THRESH_BINARY
        )

        return threshold, image


# ------------------------------------------------------------------------------------------------ #
class ThresholdISOData(Threshold):
    """Wrapper for Threshold Based Segmentation Method"""

    def __init__(
        self,
        name: str = "ISOData Threshold",
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name=name, global_threshold=global_threshold)

    def run(self, image: np.ndarray) -> Union[float, np.ndarray]:
        threshold = threshold_isodata(image=image, return_all=False)
        binary_mask = (image > threshold).astype("uint8") * 255
        return threshold, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdLi(Threshold):
    """Wrapper for Threshold Based Segmentation Method"""

    def __init__(
        self,
        name: str = "Minimum Cross-Entropy Threshold",
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name=name, global_threshold=global_threshold)

    def run(self, image: np.ndarray) -> Union[float, np.ndarray]:
        threshold = threshold_li(image=image)
        binary_mask = (image > threshold).astype("uint8") * 255
        return threshold, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdYen(Threshold):
    """Wrapper for Threshold Based Segmentation Method"""

    def __init__(
        self,
        name: str = "Yen's Multilevel Threshold",
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name=name, global_threshold=global_threshold)

    def run(self, image: np.ndarray) -> Union[float, np.ndarray]:
        threshold = threshold_yen(image=image)
        binary_mask = (image > threshold).astype("uint8") * 255
        return threshold, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdOTSU(Threshold):
    """Wrapper for Threshold Based Segmentation Method"""

    def __init__(
        self,
        name: str = "OTSU's Threshold",
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name=name, global_threshold=global_threshold)

    def run(self, image: np.ndarray) -> Union[float, np.ndarray]:
        threshold, binary_mask = cv2.threshold(
            image, 0, np.max(image), cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return threshold, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdTriangle(Threshold):
    """Wrapper for Threshold Based Segmentation Method"""

    def __init__(
        self,
        name: str = "Triangle Threshold",
        maxval: int = 255,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name=name, global_threshold=global_threshold)
        self._maxval = maxval

    def run(self, image: np.ndarray) -> Union[float, np.ndarray]:
        threshold, binary_mask = cv2.threshold(
            image, 0, self._maxval, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )
        return threshold, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdAdaptiveMean(Threshold):
    """Wrapper for Threshold Based Segmentation Method"""

    def __init__(
        self,
        name: str = "Adaptive Mean Threshold",
        global_threshold: bool = False,
        blocksize: int = 5,
        c: int = 0,
    ) -> None:
        super().__init__(name=name, global_threshold=global_threshold)
        self._blocksize = blocksize
        self._c = c

    def run(self, image: np.ndarray) -> Union[float, np.ndarray]:
        binary_mask = cv2.adaptiveThreshold(
            image,
            np.max(image),
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self._blocksize,
            self._c,
        )
        return None, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdAdaptiveGaussian(Threshold):
    """Wrapper for Threshold Based Segmentation Method"""

    def __init__(
        self,
        name: str = "Adaptive Gaussian Threshold",
        global_threshold: bool = False,
        blocksize: int = 11,
        c: int = 0,
    ) -> None:
        super().__init__(name=name, global_threshold=global_threshold)
        self._blocksize = blocksize
        self._c = c

    def run(self, image: np.ndarray) -> Union[float, np.ndarray]:
        binary_mask = cv2.adaptiveThreshold(
            image,
            np.max(image),
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self._blocksize,
            self._c,
        )
        return None, binary_mask
