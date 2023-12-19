#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/threshold.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday November 23rd 2023 12:45:30 pm                                             #
# Modified   : Tuesday December 19th 2023 10:16:21 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, arguments-differ, unused-argument, no-name-in-module, line-too-long,
# pylint: disable=invalid-name
# ------------------------------------------------------------------------------------------------ #
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_isodata, threshold_li, threshold_yen

from bcd.utils.image import convert_uint8

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------------------------------------ #
#                                    THRESHOLD ANALYZER                                            #
# ------------------------------------------------------------------------------------------------ #
class ThresholdAnalyzer:
    """Provides a visual analysis of a threshold method on an image stack.

    Args:
        show_masks (bool): Whether to show the binary masks. Default = True
        show_masked_images (bool): Whether to show masked images. Default = True
        show_histograms (bool): Whether histograms should be plotted. Default = True
    """

    __CMAP = "gray"
    __ROWHEIGHT = 3

    def __init__(
        self,
        show_masks: bool = True,
        show_masked_images: bool = True,
        show_histograms: bool = True,
    ) -> None:
        self._show_masks = show_masks
        self._show_masked_images = show_masked_images
        self._show_histograms = show_histograms

        self._images = []
        self._binary_masks = []
        self._masked_images = []
        self._histograms = []
        self._thresholds = []
        self._current_row = 0  # Used to support dynamic row plotting

        self._threshold = None

        self._axis_labels = np.array(
            [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
            ]
        )

        self._reset()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

        if self._show_histograms + self._show_masks + self._show_masked_images == 0:
            msg = "Must show masks, masked images, or histograms."
            self._logger.exception(msg)
            raise ValueError(msg)

    def analyze(self, images: tuple[np.ndarray], threshold: Threshold) -> plt.Figure:
        """Plots an analysis of the threshold methods

        Args:
            images (Tuple[np.ndarray]): Tuple of arrays containing image pixel values.
                The number of images allowed is capped at 4 for plotting purposes.
            threshold (Threshold): A Threshold object.

        """
        self._reset()
        self._threshold = threshold

        if len(images) > 0 and len(images) < 5:
            self._images = images

            self._threshold_images(images=images)
            fig = self._build_analysis_plot()
            return fig

        else:
            msg = "The length of the image tuple must be in [1,4]"
            self._logger.exception(msg)
            raise ValueError(msg)

    def _threshold_images(self, images: Tuple) -> None:
        """Compares performance of global thresholds on test images

        Args:
            images (Tuple[np.ndarray]) Tuple of test images.
        """

        for image in images:
            # Get the threshold and the binary mask
            threshold, binary_mask = self._threshold.apply_threshold(image=image)
            # Apply the mask
            masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
            # Append masks, thresholds and masked images to list.
            self._binary_masks.append(binary_mask)
            self._masked_images.append(masked_image)
            # Adaptive methods don't have global thresholds to capture.
            if threshold:
                self._thresholds.append(threshold)

    def _build_analysis_plot(self) -> None:
        """Constructs the visualization"""

        fig, axes, axis_labels = self._con_figure()

        self._plot_original_images(axes=axes, axis_labels=axis_labels)
        if self._show_masks:
            self._plot_binary_masks(axes=axes, axis_labels=axis_labels)
        if self._show_masked_images:
            self._plot_masked_images(axes=axes, axis_labels=axis_labels)
        if self._show_histograms:
            self._plot_histograms(axes=axes, axis_labels=axis_labels)

        title = self._threshold.name + " Analysis"
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()

        return fig

    def _plot_original_images(self, axes: list, axis_labels: list) -> None:
        """Plots the original images"""

        for i, img in enumerate(self._images):
            axes[self._current_row, i].imshow(img, cmap=self.__CMAP, aspect="auto")
            axes[self._current_row, i].set_xlabel(
                f"({axis_labels[self._current_row,i]}) Original Image", fontsize=8
            )
            axes[self._current_row, i].set_xticks([])
            axes[self._current_row, i].set_yticks([])

        self._current_row += 1

    def _plot_binary_masks(self, axes: list, axis_labels: list) -> None:
        """Plots the binary masks."""

        for i, mask in enumerate(self._binary_masks):
            if self._threshold.is_global:
                label = f"({axis_labels[self._current_row,i]}) {self._threshold.name} Mask, T={np.round(self._thresholds[i],0)}"
            else:
                label = (
                    f"({axis_labels[self._current_row,i]}) {self._threshold.name} Mask"
                )
            axes[self._current_row, i].imshow(mask, cmap=self.__CMAP, aspect="auto")
            axes[self._current_row, i].set_xlabel(label, fontsize=8)

            axes[self._current_row, i].set_xticks([])
            axes[self._current_row, i].set_yticks([])

        self._current_row += 1

    def _plot_masked_images(self, axes: list, axis_labels: list) -> None:
        """Plots masked images."""

        for i, img in enumerate(self._masked_images):
            if self._threshold.is_global:
                label = f"({axis_labels[self._current_row,i]}) {self._threshold.name} Output, T={np.round(self._thresholds[i],0)}"
            else:
                label = f"({axis_labels[self._current_row,i]}) {self._threshold.name} Output"
            axes[self._current_row, i].imshow(img, cmap=self.__CMAP, aspect="auto")
            axes[self._current_row, i].set_xlabel(label, fontsize=8)

            axes[self._current_row, i].set_xticks([])
            axes[self._current_row, i].set_yticks([])

        self._current_row += 1

    def _plot_histograms(self, axes: list, axis_labels: list) -> None:
        """Plot histograms with threshold annotations."""

        for i, img in enumerate(self._images):
            label = f"({axis_labels[self._current_row,i]}) {self._threshold.name} Histogram, T={np.round(self._thresholds[i],0)}"
            img = convert_uint8(img)
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            axes[self._current_row, i].plot(hist)
            axes[self._current_row, i].set_xlabel(label, fontsize=8)
            axes[self._current_row, i].axvline(
                x=self._thresholds[i],
                color="r",
                label=f"T={np.round(self._thresholds[i],0)}",
            )

            axes[self._current_row, i].set_yticks([])

    def _reset(self) -> None:
        """Resets the image the Analyzer object"""

        self._images = []
        self._binary_masks = []
        self._masked_images = []
        self._histograms = []
        self._thresholds = []
        self._current_row = 0  # Used to support dynamic row plotting

        self._threshold = None

    def _con_figure(self) -> Tuple[plt.Figure, list, np.ndarray]:
        """Configures the figure, exes, and labels"""
        self._show_histograms = self._show_histograms and self._threshold.is_global
        nrows = 1 + self._show_masks + self._show_masked_images + self._show_histograms
        ncols = len(self._images)
        nplots = nrows * ncols

        width = 12
        height = nrows * self.__ROWHEIGHT

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))

        axis_labels = self._axis_labels[:nplots]
        axis_labels = np.reshape(axis_labels, newshape=(nrows, ncols))

        return fig, axes, axis_labels


# ------------------------------------------------------------------------------------------------ #
#                                TRIANGLE THRESHOLD ANALYZER                                       #
# ------------------------------------------------------------------------------------------------ #
class ThresholdTriangleAnalyzer(ThresholdAnalyzer):
    """Triangle Threshold Analyzer

    Args:
        show_masks (bool): Whether to show the binary masks. Default = True
        show_masked_images (bool): Whether to show masked images. Default = True
        show_histograms (bool): Whether histograms should be plotted. Default = True
    """

    def __init__(
        self,
        show_masks: bool = True,
        show_masked_images: bool = True,
        show_histograms: bool = True,
    ) -> None:
        super().__init__(
            show_masks=show_masks,
            show_masked_images=show_masked_images,
            show_histograms=show_histograms,
        )

    def _plot_histograms(self, axes: list, axis_labels: list) -> None:
        """Plot histograms with threshold annotations."""

        for i, img in enumerate(self._images):
            self._plot_triangle_histogram(
                idx=i, img=img, axes=axes, axis_labels=axis_labels
            )

    def _plot_triangle_histogram(
        self, idx: int, img: np.ndarray, axes: list, axis_labels: list
    ) -> None:
        """Plots the histogram for the triangle threshold method."""

        # Source: https://bioimagebook.github.io/chapters/2-processing/3-thresholding/thresholding.html

        bins = np.arange(0, 256)

        # Create a histogram, identify the peaks and normalize counts
        # Note: We assume that the peak is on the left and the threshold is to the right.
        hist, bin_edges = np.histogram(img.ravel(), bins=bins)
        peak_ind = np.argmax(hist)
        peak_height = hist[peak_ind]
        hist = hist / peak_height

        # Identify bin centers and find last bin with non-zero count
        centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        _, ind_high = np.where(hist > 0)[0][[0, -1]]

        # Compute 'width' of the triangle (base length)
        # Normalize centers so width becomes 1
        width = centers[ind_high] - centers[peak_ind]
        centers = centers / width

        # Plot normalized histogram
        _ = axes[self._current_row, idx].hist(
            centers, bins=len(hist), weights=hist, color=(0.1, 0.1, 0.2, 0.6)
        )

        # Plot from peak to base
        x1 = centers[peak_ind]
        y1 = hist[peak_ind]
        x2 = centers[-1]
        y2 = hist[-1]
        _ = axes[self._current_row, idx].plot([x1, x2], [y1, y2])

        # Plot from threshold to peak line
        x3 = centers[int(self._thresholds[idx])]
        y3 = hist[int(self._thresholds[idx])]
        n = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        x4 = (y2 - y1) / n
        y4 = -(x2 - x1) / n

        # Find intersection
        # Thank you, wikipedia
        # https://en.wikipedia.org/wiki/Line–line_intersection#Given_two_points_on_each_line
        D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / D
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / D
        _ = axes[self._current_row, idx].plot([x3, px], [y3, py])

        # Plot the label
        label = f"({axis_labels[self._current_row,idx]}) {self._threshold.name} Histogram, T={np.round(self._thresholds[idx],0)}"
        _ = axes[self._current_row, idx].set_xlabel(label, fontsize=8)


# ------------------------------------------------------------------------------------------------ #
#                                         THRESHOLD                                                #
# ------------------------------------------------------------------------------------------------ #
class Threshold(ABC):
    """Base class for Thresholding methods.

    Args:
        name (str): Name of the threshold, defined in subclasses
        global_threshold (bool): Whether the threshold is global.
    """

    def __init__(self, name: str, global_threshold: bool) -> None:
        self._name = name
        self._global_threshold = global_threshold

    @property
    def is_global(self) -> str:
        return self._global_threshold

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def apply_threshold(self, image: np.ndarray) -> Union[float, np.array]:
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

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
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

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
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

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
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

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
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

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
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
        global_threshold: bool = True,
    ) -> None:
        super().__init__(name=name, global_threshold=global_threshold)

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
        threshold, binary_mask = cv2.threshold(
            image, 0, np.max(image), cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
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

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
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

    def apply_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
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
class ThresholdSurveyor:
    """Surveys multiple thresholds on an image."""

    def __init__(self) -> None:
        self._methods = {
            "triangle": ThresholdTriangle,
            "isodata": ThresholdISOData,
            "otsu": ThresholdOTSU,
            "li": ThresholdLi,
            "yen": ThresholdYen,
            "local_gaussian": ThresholdAdaptiveGaussian,
            "local_mean": ThresholdAdaptiveMean,
        }
        self._axis_idx = np.array(
            ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        )
        self._images = []
        self._labels = []

    def tryall(self, image: np.ndarray) -> None:
        """Plots image segemntation results for all methods

        Args:
            image (np.ndarray): The image in numpy unit8 format.
        """

        label = self._axis_idx[0] + " Original Image"
        self._images.append(image)
        self._labels.append(label)

        # Iterate over the methods, and create a list of
        # segmented images.
        for idx, method in enumerate(self._methods.values(), 1):
            m = method()
            thresh, binary_mask = m.apply_threshold(image)
            segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)
            label = self._axis_idx[idx] + " " + m.name
            if thresh:
                label += " T=" + str(int(thresh))
            self._images.append(segmented_image)
            self._labels.append(label)

        nrows = 2
        ncols = 4
        width = 12
        height = 3 * nrows

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
        axes = axes.flatten()

        # Iterate over images and labels and plot on axes.
        for idx, image in enumerate(self._images):
            axes[idx].imshow(image, cmap="gray", aspect="auto")
            axes[idx].set_xlabel(self._labels[idx])
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

        plt.tight_layout()
        return fig
