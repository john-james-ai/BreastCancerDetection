#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/denoise/analyze.py                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday November 23rd 2023 12:45:30 pm                                             #
# Modified   : Thursday November 23rd 2023 06:41:41 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
import string
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.util import random_noise

from bcd.utils.image import convert_uint8


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, arguments-differ, unused-argument
# ------------------------------------------------------------------------------------------------ #
class DenoiserAnalyzer(ABC):
    """Analyzes a Denoiser

    Args:
        denoiser (str): The name of the denoiser.
    """

    __FP_IMAGE = "jbook/figures/mammogram.png"
    __CMAP = "gray"

    def __init__(self, denoiser: str) -> None:
        self._denoiser = denoiser
        self._image_ground_truth = cv2.imread(self.__FP_IMAGE, cv2.IMREAD_GRAYSCALE)
        self._image_ground_truth_pil = Image.open(self.__FP_IMAGE)
        self._image_degraded = None
        self._image_filtered = None
        self._image_noise = None

        self._hist_ground_truth = None
        self._hist_degraded = None
        self._hist_filtered = None
        self._hist_noise = None

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def apply_filter(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Applies a filter to an image and returns the filtered image.."""

    def compare(self, *args, sizes: tuple = (3, 5, 9, 13), **kwargs) -> None:
        """Compares performance of n filters

        Args:
            sizes (tuple): Tuple of kernel sizes
            args (list): List of arguments for subclasses.
            kwargs (dict) Dictionary of arguments for subclasses.
        """
        if self._image_degraded is None:
            msg = "Image must be degraded before filtering can be applied."
            self._logger.exception(msg)

        images = {}
        for size in sizes:
            images[size] = self.apply_filter(image=self._image_degraded, size=size)
        return self._plot_filtered_images(images=images)

    def add_gaussian_noise(self, var: float = 0.1) -> None:
        """Adds gaussian noise to the image."""
        self._image_degraded = random_noise(
            self._image_ground_truth, mode="gaussian", var=var
        )
        self._image_degraded = convert_uint8(self._image_degraded)

    def add_poisson_noise(self) -> None:
        """Adds poisson noise to the image."""
        self._image_degraded = random_noise(self._image_ground_truth, mode="poisson")
        self._image_degraded = convert_uint8(self._image_degraded)

    def add_snp_noise(self, amount: float = 0.3) -> None:
        """Adds salt and pepper noise to the image."""
        self._image_degraded = random_noise(
            self._image_ground_truth, mode="s&p", amount=amount
        )
        self._image_degraded = convert_uint8(self._image_degraded)

    def add_speckle_noise(self, var: float = 0.1) -> None:
        """Adds speckle noise to the image."""
        self._image_degraded = random_noise(
            self._image_ground_truth, mode="speckle", var=var
        )
        self._image_degraded = convert_uint8(self._image_degraded)

    def add_quantize_noise(self, colors: int = 2) -> None:
        """Quantizes an image to a given number of colors."""
        self._image_degraded = self._image_ground_truth_pil.quantize(colors=colors)
        self._image_degraded = np.array(self._image_degraded)
        self._image_degraded = convert_uint8(self._image_degraded, invert=True)

    def analyze(self, *args, **kwargs) -> plt.Figure:
        """Plots an analysis of filter performance

        Plots two rows of subplots. The first row contains
        the original, degraded, filtered, and noise image.
        The second row contains the histograms for those images.

        Args:
            *args, **kwargs: Arguments passed to the subclass apply_filter method.
        """

        fig = plt.figure(figsize=(12, 6), dpi=200)

        self._image_filtered = self.apply_filter(
            image=self._image_degraded, *args, **kwargs
        )

        self._image_noise = self._image_degraded - self._image_filtered

        self._create_histograms()

        return self._build_analysis_plot(fig=fig)

    def _create_histograms(self) -> None:
        """Creates the four histograms"""
        self._hist_ground_truth = cv2.calcHist(
            [self._image_ground_truth], [0], None, [256], [0, 256]
        )
        self._hist_degraded = cv2.calcHist(
            [self._image_degraded], [0], None, [256], [0, 256]
        )
        self._hist_filtered = cv2.calcHist(
            [self._image_filtered], [0], None, [256], [0, 256]
        )
        self._hist_noise = cv2.calcHist([self._image_noise], [0], None, [256], [0, 256])

    def _plot_filtered_images(self, images: dict) -> plt.Figure:
        """Plots the filtered images"""
        pfx = ["(a)", "(b)", "(c)", "(d)"]
        fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)
        axes = axes.flatten()

        for idx, (size, image) in enumerate(images.items()):
            label = f"{pfx[idx]} {string.capwords(self._denoiser)} {str(size)}x{str(size)} Kernel"
            axes[idx].imshow(image, cmap=self.__CMAP)
            axes[idx].set_xlabel(label, fontsize=10)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

        title = string.capwords(s=self._denoiser) + " Performance Characteristics"
        fig.suptitle(title, fontsize=12)

        plt.tight_layout()

        return fig

    def _build_analysis_plot(self, fig: plt.Figure) -> plt.Figure:
        """Constructs the plot on the designated Figure object and returns it, without display."""
        spec = fig.add_gridspec(ncols=4, nrows=2)

        # Plot the images
        ax0 = fig.add_subplot(spec[0, 0])
        ax0.imshow(self._image_ground_truth, cmap=self.__CMAP)
        ax0.set_xlabel("(a) Original Image", fontsize=10)
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = fig.add_subplot(spec[0, 1])
        ax1.imshow(self._image_degraded, cmap=self.__CMAP)
        ax1.set_xlabel("(b) Degraded Image", fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(spec[0, 2])
        ax2.imshow(self._image_filtered, cmap=self.__CMAP)
        ax2.set_xlabel("(c) Filtered Image", fontsize=10)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = fig.add_subplot(spec[0, 3])
        ax3.imshow(self._image_noise, cmap=self.__CMAP)
        ax3.set_xlabel("(d) Noise", fontsize=10)
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Plot Histograms
        ax4 = fig.add_subplot(spec[1, 0])
        ax4.plot(self._hist_ground_truth)
        ax4.set_xlabel("(e) Original Image Histogram", fontsize=10)
        ax4.set_yticks([])

        ax5 = fig.add_subplot(spec[1, 1])
        ax5.plot(self._hist_degraded)
        ax5.set_xlabel("(f) Degraded Image Histogram", fontsize=10)
        ax5.set_yticks([])

        ax6 = fig.add_subplot(spec[1, 2])
        ax6.plot(self._hist_filtered)
        ax6.set_xlabel("(g) Filtered Image Histogram", fontsize=10)
        ax6.set_yticks([])

        ax7 = fig.add_subplot(spec[1, 3])
        ax7.plot(self._hist_noise)
        ax7.set_xlabel("(f) Degraded Image Histogram", fontsize=10)
        ax7.set_yticks([])

        title = string.capwords(s=self._denoiser) + " Performance Analysis"
        plt.tight_layout()
        fig.suptitle(title, fontsize=12)


# ------------------------------------------------------------------------------------------------ #
#                                    MEAN FILTER VISUALIZER                                        #
# ------------------------------------------------------------------------------------------------ #
class MeanFilterAnalyzer(DenoiserAnalyzer):
    """Analyzes Mean Filters"""

    def __init__(self, denoiser: str = "Mean Filter") -> None:
        super().__init__(denoiser)

    def apply_filter(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        return cv2.blur(image, (size, size))


# ------------------------------------------------------------------------------------------------ #
#                                GAUSSIAN FILTER VISUALIZER                                        #
# ------------------------------------------------------------------------------------------------ #
class GaussianFilterAnalyzer(DenoiserAnalyzer):
    """Analyzes Gaussian Filters"""

    def __init__(self, denoiser: str = "Gaussian Filter") -> None:
        super().__init__(denoiser)

    def apply_filter(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        return cv2.GaussianBlur(image, (size, size), 0)


# ------------------------------------------------------------------------------------------------ #
#                                MEDIAN FILTER VISUALIZER                                          #
# ------------------------------------------------------------------------------------------------ #
class MedianFilterAnalyzer(DenoiserAnalyzer):
    """Analyzes Gaussian Filters"""

    def __init__(self, denoiser: str = "Median Filter") -> None:
        super().__init__(denoiser)

    def apply_filter(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        return cv2.medianBlur(image, size)
