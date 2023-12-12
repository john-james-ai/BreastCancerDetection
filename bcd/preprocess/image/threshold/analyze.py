#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/threshold/analyze.py                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday November 23rd 2023 12:45:30 pm                                             #
# Modified   : Monday December 11th 2023 10:33:09 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, arguments-differ, unused-argument, no-name-in-module
# ------------------------------------------------------------------------------------------------ #
import logging
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------------------------------------ #
class ThresholdAnalyzer(ABC):
    """Analyzes a Binary Threshold Method

    Args:
        name (str): Name of the threshold technique
        threshold (Callable): The thresholding callable.
        global_threshold (bool): True if global threshold, otherwise False.

    """

    __CMAP = "gray"
    __ROWHEIGHT = 3

    def __init__(
        self, name: str, threshold: Callable, global_threshold: bool = True
    ) -> None:
        self._name = name
        self._threshold = threshold
        self._global_threshold = global_threshold

        self._images = []
        self._binary_masks = []
        self._histograms = []
        self._thresholds = []

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        """Applies a threshold and returns the threshold and the binary mask.

        Args:
            image (np.ndarray): Numpy array of test image

        Returns:
            threshold (float): The threshold applied. Only for global threshold methods
            binary_mask (np.ndarray): Binary mask of image.
        """

    def analyze(self, images: np.ndarray, *args, **kwargs) -> plt.Figure:
        """Plots an analysis of the threshold methods

        Args:
            images (Tuple[np.ndarray]): Tuple of arrays containing image pixel values.

        """
        self._images = images

        if self._global_threshold:
            self._compare_global_thresholds(images=images, *args, **kwargs)
            self._build_global_analysis_plot()
        else:
            self._compare_adaptive_thresholds(images=images, *args, **kwargs)
            self._build_adaptive_analysis_plot()

    def _compare_global_thresholds(self, images: Tuple, *args, **kwargs) -> None:
        """Compares performance of global thresholds on test images

        Args:
            images (Tuple[np.ndarray]) Tuple of test images.
        """

        for image in images:
            threshold, binary_mask = self.apply_threshold(image=image, *args, **kwargs)
            histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
            self._binary_masks.append(binary_mask)
            self._thresholds.append(threshold)
            self._histograms.append(histogram)

    def _compare_adaptive_thresholds(self, images: Tuple, *args, **kwargs) -> None:
        """Compares performance of adaptive thresholds on test images

        Args:
            images (Tuple[np.ndarray]) Tuple of test images.
        """

        for image in images:
            binary_mask = self.apply_threshold(image=image, *args, **kwargs)
            self._binary_masks.append(binary_mask)

    def _build_global_analysis_plot(self) -> None:
        height = (
            3 * self.__ROWHEIGHT if self._global_threshold else 2 * self.__ROWHEIGHT
        )

        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, height))
        labels = np.array(["a", "b", "c", "d"])

        # Plot images
        for i, img in enumerate(self._images):
            axes[0, i].imshow(img, cmap=self.__CMAP, aspect="auto")
            axes[0, i].set_xlabel(f"Original Image ({labels[i]})", fontsize=10)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

        # Plot binary masks
        for i, mask in enumerate(self._binary_masks):
            axes[1, i].imshow(mask, cmap=self.__CMAP, aspect="auto")
            axes[1, i].set_xlabel(f"{self._name} ({labels[i]})", fontsize=10)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

        # Plot histograms
        for i, hist in enumerate(self._histograms):
            axes[2, i].plot(hist)
            axes[2, i].axvline(
                x=self._thresholds[i],
                color="r",
                label=f"T={round(self._thresholds[i],1)}",
            )
            axes[2, i].set_xlabel(f"Histogram ({labels[i]})", fontsize=10)
            axes[2, i].legend(loc="upper right")
            # axes[2, i].set_xticks([])
            # axes[2, i].set_yticks([])

        title = self._name + " Analysis"
        fig.suptitle(title, fontsize=12)

        plt.tight_layout()
        return fig

    def _build_adaptive_analysis_plot(self) -> None:
        height = 2 * self.__ROWHEIGHT

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, height))
        labels = np.array(["a", "b", "c", "d"])
        # Plot images
        for i, img in enumerate(self._images):
            axes[0, i].imshow(img, cmap=self.__CMAP, aspect="auto")
            axes[0, i].set_xlabel(f"Original Image ({labels[i]})", fontsize=10)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

        # Plot binary masks
        for i, mask in enumerate(self._binary_masks):
            axes[1, i].imshow(mask, cmap=self.__CMAP, aspect="auto")
            axes[1, i].set_xlabel(f"{self._name} ({labels[i]})", fontsize=10)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

        title = self._name + " Analysis"
        fig.suptitle(title, fontsize=12)

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------------------------------------ #
class ThresholdMean(ThresholdAnalyzer):
    """Plots the threshold analysis for the Mean threshold method"""

    def __init__(
        self,
        name: str = "Mean Threshold",
        threshold: Callable = threshold_mean,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, threshold, global_threshold)

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        threshold = self._threshold(image=image)
        image = image > threshold
        return threshold, image


# ------------------------------------------------------------------------------------------------ #
class ThresholdISOData(ThresholdAnalyzer):
    """Plots the threshold analysis for the ISOData threshold method"""

    def __init__(
        self,
        name: str = "ISO Data Threshold",
        threshold: Callable = threshold_isodata,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, threshold, global_threshold)

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        threshold = self._threshold(image=image)
        image = image > threshold
        return threshold, image


# ------------------------------------------------------------------------------------------------ #
class ThresholdMinimum(ThresholdAnalyzer):
    """Plots the threshold analysis for minimum threshold method"""

    def __init__(
        self,
        name: str = "Minimum Threshold",
        threshold: Callable = threshold_minimum,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, threshold, global_threshold)

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        threshold = self._threshold(image=image)
        image = image > threshold
        return threshold, image


# ------------------------------------------------------------------------------------------------ #
class ThresholdLi(ThresholdAnalyzer):
    """Plots the threshold analysis for Li threshold method"""

    def __init__(
        self,
        name: str = "Li Threshold",
        threshold: Callable = threshold_li,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, threshold, global_threshold)

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        threshold = self._threshold(image=image)
        image = image > threshold
        return threshold, image


# ------------------------------------------------------------------------------------------------ #
class ThresholdYen(ThresholdAnalyzer):
    """Plots the threshold analysis for Yen threshold method"""

    def __init__(
        self,
        name: str = "Yen Threshold",
        threshold: Callable = threshold_yen,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, threshold, global_threshold)

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        threshold = self._threshold(image=image)
        image = image > threshold
        return threshold, image


# ------------------------------------------------------------------------------------------------ #
class ThresholdTriangle(ThresholdAnalyzer):
    """Plots the threshold analysis for Yen threshold method"""

    def __init__(
        self,
        name: str = "Triangle Threshold",
        threshold: Callable = threshold_triangle,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, threshold, global_threshold)

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        threshold = self._threshold(image=image)
        image = image > threshold
        return threshold, image


# ------------------------------------------------------------------------------------------------ #
class ThresholdOTSU(ThresholdAnalyzer):
    """Plots the threshold analysis for OTUS's threshold method"""

    def __init__(
        self,
        name: str = "OTSU's Threshold",
        threshold: Callable = threshold_otsu,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, threshold, global_threshold)

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        threshold = self._threshold(image=image)
        image = image > threshold
        return threshold, image


# ------------------------------------------------------------------------------------------------ #
class ThresholdAdaptiveMean(ThresholdAnalyzer):
    """Plots the threshold analysis for adaptive mean threshold method"""

    def __init__(
        self,
        name: str = "Adaptive Mean Threshold",
        threshold: Callable = cv2.adaptiveThreshold,
        global_threshold: bool = False,
    ) -> None:
        super().__init__(name, threshold, global_threshold)

    def apply_threshold(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._threshold(
            src=image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            *args,
            **kwargs,
        )
