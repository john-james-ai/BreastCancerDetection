#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/analysis/threshold.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday November 23rd 2023 12:45:30 pm                                             #
# Modified   : Thursday December 14th 2023 11:19:04 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, arguments-differ, unused-argument, no-name-in-module
# ------------------------------------------------------------------------------------------------ #
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage.filters import threshold_isodata, threshold_li

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

    def __init__(self, name: str, global_threshold: bool = True) -> None:
        self._name = name
        self._global_threshold = global_threshold

        self._images = []
        self._binary_masks = []
        self._masked_images = []
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

    def analyze(self, images: np.ndarray) -> plt.Figure:
        """Plots an analysis of the threshold methods

        Args:
            images (Tuple[np.ndarray]): Tuple of arrays containing image pixel values.

        """
        self._images = images

        self._compare_thresholds(images=images)
        fig = self._build_analysis_plot()
        return fig

    def _compare_thresholds(self, images: Tuple) -> None:
        """Compares performance of global thresholds on test images

        Args:
            images (Tuple[np.ndarray]) Tuple of test images.
        """

        for image in images:
            # Get the threshold and the binary mask
            threshold, binary_mask = self.apply_threshold(image=image)
            # Apply the mask
            masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
            # Append masks, thresholds and masked images to list.
            self._binary_masks.append(binary_mask)
            self._masked_images.append(masked_image)
            if threshold:
                self._thresholds.append(threshold)

    def _build_analysis_plot(self) -> None:
        nrows = 3
        ncols = 4
        height = nrows * self.__ROWHEIGHT

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, height))
        labels = np.array(
            [["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j", "k", "l"]]
        )

        # Plot images
        for i, img in enumerate(self._images):
            axes[0, i].imshow(img, cmap=self.__CMAP, aspect="auto")
            axes[0, i].set_xlabel(f"({labels[0,i]}) Original Image", fontsize=10)

        # Plot binary masks
        for i, mask in enumerate(self._binary_masks):
            if self._global_threshold:
                label = f"({labels[1,i]}) {self._name} Mask, T={np.round(self._thresholds[i],0)}"
            else:
                label = f"({labels[1,i]}) {self._name} Mask"
            axes[1, i].imshow(mask, cmap=self.__CMAP, aspect="auto")
            axes[1, i].set_xlabel(label, fontsize=10)

        # Plot masked images
        for i, img in enumerate(self._masked_images):
            if self._global_threshold:
                label = f"({labels[2,i]}) {self._name} Output, T={np.round(self._thresholds[i],0)}"
            else:
                label = f"({labels[2,i]}) {self._name} Output"
            axes[2, i].imshow(img, cmap=self.__CMAP, aspect="auto")
            axes[2, i].set_xlabel(label, fontsize=10)

        # Remove ticks
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        title = self._name + " Analysis"
        fig.suptitle(title, fontsize=12)

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------------------------------------ #
class ThresholdManual(ThresholdAnalyzer):
    """Plots the threshold analysis for the Mean threshold method"""

    def __init__(
        self,
        name: str = "Manual Threshold",
        threshold: float = 0.1,
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, global_threshold)
        self._threshold = threshold

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
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
class ThresholdISOData(ThresholdAnalyzer):
    """Plots the threshold analysis for the Mean threshold method"""

    def __init__(
        self,
        name: str = "ISOData Threshold",
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, global_threshold)

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
        threshold = threshold_isodata(image=image, return_all=False)
        binary_mask = (image > threshold).astype("uint8") * 255
        return threshold, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdLi(ThresholdAnalyzer):
    """Plots the threshold analysis for the Mean threshold method"""

    def __init__(
        self,
        name: str = "Li's Minimum Cross-Entropy Threshold",
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, global_threshold)

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
        threshold = threshold_li(image=image)
        binary_mask = (image > threshold).astype("uint8") * 255
        return threshold, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdOTSU(ThresholdAnalyzer):
    """Plots the threshold analysis for OTUS's threshold method"""

    def __init__(
        self,
        name: str = "OTSU's Threshold",
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, global_threshold)

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        threshold, binary_mask = cv2.threshold(
            image, 0, np.max(image), cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return threshold, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdTriangle(ThresholdAnalyzer):
    """Plots the threshold analysis for the Triangle threshold method"""

    def __init__(
        self,
        name: str = "Triangle Threshold",
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name, global_threshold)

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        threshold, binary_mask = cv2.threshold(
            image, 0, np.max(image), cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )
        return threshold, binary_mask


# ------------------------------------------------------------------------------------------------ #
class ThresholdAdaptiveMean(ThresholdAnalyzer):
    """Plots the threshold analysis for adaptive mean threshold method"""

    def __init__(
        self,
        name: str = "Adaptive Mean Threshold",
        global_threshold: bool = False,
        blocksize: int = 11,
        c: int = 0,
    ) -> None:
        super().__init__(name, global_threshold)
        self._blocksize = blocksize
        self._c = c

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
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
class ThresholdAdaptiveGaussian(ThresholdAnalyzer):
    """Plots the threshold analysis for adaptive mean threshold method"""

    def __init__(
        self,
        name: str = "Adaptive Gaussian Threshold",
        global_threshold: bool = False,
        blocksize: int = 11,
        c: int = 0,
    ) -> None:
        super().__init__(name, global_threshold)
        self._blocksize = blocksize
        self._c = c

    def apply_threshold(
        self, image: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:
        binary_mask = cv2.adaptiveThreshold(
            image,
            np.max(image),
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self._blocksize,
            self._c,
        )
        return None, binary_mask


# ------------------------------------------------------------------------------------------------ #
class TryAllThresholds:
    """Plots threshold masks or images for all supported methods.

    Args:
        threshold (Union[float,int]): Manual threshold value. Default = 0.1
        blocksize (int): Blocksize for adaptive thresholds
        c (int): C constant for adaptive thresholds.
    """

    def __init__(self, threshold: float = 0.1, blocksize: int = 11, c: int = 0) -> None:
        self._threshold = threshold
        self._blocksize = blocksize
        self._c = c

        self._images = []
        self._labels = []

        self._show_masks = False

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def analyze(self, image: np.ndarray, show_masks: bool = False) -> plt.Figure:
        self._images.append(image)
        self._show_masks = show_masks
        self._labels.append("(a) Original Image")

        self._threshold_isodata(image=image)
        self._threshold_triangle(image=image)
        self._threshold_li(image=image)
        self._threshold_otsu(image=image)
        self._threshold_adaptive_mean(image=image)
        self._threshold_adaptive_gaussian(image=image)

        self._plot_results()

    def _threshold_isodata(self, image: np.ndarray) -> None:
        analyzer = ThresholdISOData()
        thresh, img = self._apply_threshold(analyzer=analyzer, image=image)
        label = f"(c) ISOData Threshold, T={round(thresh,0)}"
        self._images.append(img)
        self._labels.append(label)

    def _threshold_triangle(self, image: np.ndarray) -> None:
        analyzer = ThresholdTriangle()
        thresh, img = self._apply_threshold(analyzer=analyzer, image=image)
        label = f"(d) Triangle Threshold, T={round(thresh,0)}"
        self._images.append(img)
        self._labels.append(label)

    def _threshold_li(self, image: np.ndarray) -> None:
        analyzer = ThresholdLi()
        thresh, img = self._apply_threshold(analyzer=analyzer, image=image)
        label = f"(e) Li's Minimum Cross-Entropy Threshold\nT={round(thresh,0)}"
        self._images.append(img)
        self._labels.append(label)

    def _threshold_otsu(self, image: np.ndarray) -> None:
        analyzer = ThresholdOTSU()
        thresh, img = self._apply_threshold(analyzer=analyzer, image=image)
        label = f"(f) OTSU's Threshold, T={round(thresh,0)}"
        self._images.append(img)
        self._labels.append(label)

    def _threshold_adaptive_mean(self, image: np.ndarray) -> None:
        analyzer = ThresholdAdaptiveMean(blocksize=self._blocksize, c=self._c)
        _, img = self._apply_threshold(analyzer=analyzer, image=image)
        label = f"(g) Adaptive Mean Threshold\nblocksize={self._blocksize}, C={self._c}"
        self._images.append(img)
        self._labels.append(label)

    def _threshold_adaptive_gaussian(self, image: np.ndarray) -> None:
        analyzer = ThresholdAdaptiveGaussian(blocksize=self._blocksize, c=self._c)
        _, img = self._apply_threshold(analyzer=analyzer, image=image)

        label = (
            f"(h) Adaptive Gaussian Threshold\nblocksize={self._blocksize}, C={self._c}"
        )
        self._images.append(img)
        self._labels.append(label)

    def _apply_threshold(
        self, analyzer: ThresholdAnalyzer, image: np.ndarray
    ) -> np.ndarray:
        """Applies the threshold and returns the mask or the thresholded image"""
        thresh, binary_mask = analyzer.apply_threshold(image=image)

        if self._show_masks:
            return thresh, binary_mask
        else:
            masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
            return thresh, masked_image

    def _plot_results(self) -> plt.Figure:
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(3, 3, figure=fig)

        # Show original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self._images[0], cmap="gray", aspect="auto")
        ax1.set_xlabel(self._labels[0], fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Show histogram of original image
        histogram = cv2.calcHist([self._images[0]], [0], None, [256], [0, 256])
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.plot(histogram)
        ax2.set_xlabel("(b) Histogram Original Image", fontsize=10)
        ax2.set_yticks([])

        # Plot Threshold Results
        method_no = 0
        for i in range(2):
            for j in range(3):
                method_no += 1  # Already plotted image 0 (original image)
                ax = fig.add_subplot(gs[i + 1, j])
                ax.imshow(self._images[method_no], cmap="gray", aspect="auto")
                ax.set_xlabel(self._labels[method_no], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        # fig.suptitle("Threshold Segmentation Analysis", fontsize=12)
        return fig
