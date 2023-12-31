#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/explore/methods/noise.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday November 23rd 2023 12:45:30 pm                                             #
# Modified   : Monday December 25th 2023 11:49:55 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, arguments-differ, unused-argument, no-name-in-module
# ------------------------------------------------------------------------------------------------ #
import logging
import string
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image
from skimage.restoration import denoise_wavelet
from skimage.util import random_noise

from bcd.utils.image import grayscale

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)


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
        self._image_degraded = grayscale(self._image_degraded)

    def add_poisson_noise(self) -> None:
        """Adds poisson noise to the image."""
        self._image_degraded = random_noise(self._image_ground_truth, mode="poisson")
        self._image_degraded = grayscale(self._image_degraded)

    def add_snp_noise(self, amount: float = 0.3) -> None:
        """Adds salt and pepper noise to the image."""
        self._image_degraded = random_noise(
            self._image_ground_truth, mode="s&p", amount=amount
        )
        self._image_degraded = grayscale(self._image_degraded)

    def add_speckle_noise(self, var: float = 0.1) -> None:
        """Adds speckle noise to the image."""
        self._image_degraded = random_noise(
            self._image_ground_truth, mode="speckle", var=var
        )
        self._image_degraded = grayscale(self._image_degraded)

    def add_quantize_noise(self, colors: int = 2) -> None:
        """Quantizes an image to a given number of colors."""
        self._image_degraded = self._image_ground_truth_pil.quantize(colors=colors)
        self._image_degraded = np.array(self._image_degraded)
        self._image_degraded = grayscale(self._image_degraded, invert=True)

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

        title = string.capwords(s=self._denoiser) + " Performance Analysis"
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
        ax7.set_xlabel("(h) Degraded Image Histogram", fontsize=10)
        ax7.set_yticks([])

        title = string.capwords(s=self._denoiser) + " Performance Characteristics"
        plt.tight_layout()
        fig.suptitle(title, fontsize=12)

        return fig


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
    """Analyzes Median Filters"""

    def __init__(self, denoiser: str = "Median Filter") -> None:
        super().__init__(denoiser)

    def apply_filter(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        return cv2.medianBlur(image, size)


# ------------------------------------------------------------------------------------------------ #
#                              BILATERAL FILTER VISUALIZER                                         #
# ------------------------------------------------------------------------------------------------ #
class BilateralFilterAnalyzer(DenoiserAnalyzer):
    """Analyzes Bilateral Filters"""

    __CMAP = "gray"

    def __init__(self, denoiser: str = "Bilateral Filter") -> None:
        super().__init__(denoiser)

    def apply_filter(
        self,
        image: np.ndarray,
        d: int = -1,
        sigma_range: float = 25,
        sigma_domain: float = 25,
    ) -> np.ndarray:
        return cv2.bilateralFilter(
            image, d=d, sigmaColor=sigma_range, sigmaSpace=sigma_domain
        )

    def compare(self, sigma: tuple = (10, 25, 75, 150)) -> None:
        """Compares performance of n filters

        Args:
            sigma (tuple): Various values for range and domain parameters.
        """
        if self._image_degraded is None:
            msg = "Image must be degraded before filtering can be applied."
            self._logger.exception(msg)

        images = {}
        for s in sigma:
            images[s] = self.apply_filter(
                image=self._image_degraded, sigma_range=s, sigma_domain=s
            )
        return self._plot_filtered_images(images=images)

    def _plot_filtered_images(self, images: dict) -> plt.Figure:
        """Plots the filtered images"""
        pfx = ["(a)", "(b)", "(c)", "(d)"]
        fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)
        axes = axes.flatten()

        for idx, (s, image) in enumerate(images.items()):
            label = rf"{pfx[idx]} {string.capwords(self._denoiser)} $\sigma_r$={s}, $\sigma_s$={s}."
            axes[idx].imshow(image, cmap=self.__CMAP)
            axes[idx].set_xlabel(label, fontsize=10)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

        title = string.capwords(s=self._denoiser) + " Performance Analysis"
        fig.suptitle(title, fontsize=12)

        plt.tight_layout()

        return fig


# ------------------------------------------------------------------------------------------------ #
#                              NON-LOCAL MEANS FILTER VISUALIZER                                   #
# ------------------------------------------------------------------------------------------------ #
class NLMeansFilterAnalyzer(DenoiserAnalyzer):
    """Analyzes Non-Local Means Filters"""

    __CMAP = "gray"

    def __init__(self, denoiser: str = "Non-Local Means Filter") -> None:
        super().__init__(denoiser)

    def apply_filter(
        self,
        image: np.ndarray,
        kernel_size: int = 7,
        search_window: int = 21,
        h: int = 10,
    ) -> np.ndarray:
        return cv2.fastNlMeansDenoising(
            image, templateWindowSize=kernel_size, searchWindowSize=search_window, h=h
        )

    def compare(self, h: tuple = (10, 20, 40, 80)) -> None:
        """Compares performance of various denoisers

        Args:
            h (tuple): Various values for h, which determines filter strength.
        """
        if self._image_degraded is None:
            msg = "Image must be degraded before filtering can be applied."
            self._logger.exception(msg)

        images = {}
        for h_ in h:
            images[h_] = self.apply_filter(image=self._image_degraded, h=h_)
        return self._plot_filtered_images(images=images)

    def _plot_filtered_images(self, images: dict) -> plt.Figure:
        """Plots the filtered images"""
        pfx = ["(a)", "(b)", "(c)", "(d)"]
        fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)
        axes = axes.flatten()

        for idx, (h, image) in enumerate(images.items()):
            label = rf"{pfx[idx]} {string.capwords(self._denoiser)} $h$ = {h}."
            axes[idx].imshow(image, cmap=self.__CMAP)
            axes[idx].set_xlabel(label, fontsize=10)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

        title = string.capwords(s=self._denoiser) + " Performance Analysis"
        fig.suptitle(title, fontsize=12)

        plt.tight_layout()

        return fig


# ------------------------------------------------------------------------------------------------ #
#                              BUTTERWORTH FILTER VISUALIZER                                       #
# ------------------------------------------------------------------------------------------------ #
class ButterworthFilterAnalyzer(DenoiserAnalyzer):
    """Analyzes the Butterworth Filter"""

    __CMAP = "gray"

    def __init__(self, denoiser: str = "Butterworth Filter") -> None:
        super().__init__(denoiser)

    def apply_filter(
        self,
        image: np.ndarray,
        order: int,
        cutoff_frequency: int,
        sampling_frequency: float = 44100,
    ) -> np.ndarray:
        b, a = scipy.signal.butter(
            N=order,
            Wn=cutoff_frequency,
            fs=sampling_frequency,
            btype="lowpass",
            analog=False,
        )
        return scipy.signal.filtfilt(b, a, image)
        # return grayscale(filtered_image)

    def compare(
        self,
        orders: tuple = (6, 8, 10),
        cutoff_frequencies: tuple = (500, 750, 1000, 1500),
    ) -> None:
        """Compares performance of various denoisers

        Args:
            order (tuple): Values for the order of the filter
            cutoff_frequencies (tuple): Values for cutoff frequencies.
        """
        labels = np.array(
            [
                ["(a)", "(b)", "(c)", "(d)"],
                ["(e)", "(f)", "(g)", "(h)"],
                ["(i)", "(j)", "(k)", "(l)"],
            ]
        )

        if self._image_degraded is None:
            msg = "Image must be degraded before filtering can be applied."
            self._logger.exception(msg)

        fig, axes = plt.subplots(figsize=(12, 7), nrows=3, ncols=4)

        for i, order in enumerate(orders):
            for j, cutoff in enumerate(cutoff_frequencies):
                img = self.apply_filter(
                    image=self._image_degraded,
                    order=order,
                    cutoff_frequency=cutoff,
                )
                label = f"{labels[i,j]} Order: {order} Cutoff: {cutoff}"
                _ = axes[i, j].imshow(img, cmap="gray")
                _ = axes[i, j].set_xlabel(label)
                _ = axes[i, j].set_xticks([])
                _ = axes[i, j].set_yticks([])
        plt.tight_layout()
        return fig

    def _create_histograms(self) -> None:
        """Creates the four histograms"""
        self._hist_ground_truth = cv2.calcHist(
            [self._image_ground_truth], [0], None, [256], [0, 256]
        )
        self._hist_degraded = cv2.calcHist(
            [self._image_degraded], [0], None, [256], [0, 256]
        )

        self._hist_filtered = cv2.calcHist(
            [grayscale(self._image_filtered)], [0], None, [256], [0, 256]
        )

        self._hist_noise = cv2.calcHist(
            [grayscale(self._image_noise)], [0], None, [256], [0, 256]
        )


# ------------------------------------------------------------------------------------------------ #
#                                  WAVELET FILTER VISUALIZER                                       #
# ------------------------------------------------------------------------------------------------ #
class WaveletFilterAnalyzer(DenoiserAnalyzer):
    """Analyzes Wavelet Filters"""

    def __init__(self, denoiser: str = "Wavelet Filter") -> None:
        super().__init__(denoiser)

    def apply_filter(
        self,
        image: np.ndarray,
        sigma: float = None,
        wavelet: str = "haar",
        mode: str = "soft",
        method: str = "BayesShrink",
        channel_axis: int = None,
    ) -> np.ndarray:
        return denoise_wavelet(
            image=image,
            sigma=sigma,
            wavelet=wavelet,
            mode=mode,
            method=method,
            channel_axis=channel_axis,
        )

    def _create_histograms(self) -> None:
        """Creates the four histograms"""
        self._hist_ground_truth = cv2.calcHist(
            [self._image_ground_truth], [0], None, [256], [0, 256]
        )
        self._hist_degraded = cv2.calcHist(
            [self._image_degraded], [0], None, [256], [0, 256]
        )

        self._hist_filtered = cv2.calcHist(
            [grayscale(self._image_filtered)], [0], None, [256], [0, 256]
        )

        self._hist_noise = cv2.calcHist(
            [grayscale(self._image_noise)], [0], None, [256], [0, 256]
        )

    def compare(self, *args, **kwargs) -> None:
        raise NotImplementedError
