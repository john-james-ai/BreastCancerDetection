#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/experiment/visualize.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday November 3rd 2023 06:54:34 pm                                                #
# Modified   : Saturday November 4th 2023 06:40:07 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import string
import warnings
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("Paired")

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
#                             VISUALIZE RESULTS BASE CLASS                                         #
# ------------------------------------------------------------------------------------------------ #
class VisualizeResults(ABC):
    """Visualizes the results of an experiment."""

    @abstractmethod
    def plot_results(self) -> None:
        """Plots the best results from the experiment."""

    @abstractmethod
    def plot_images(self, metric: str) -> None:
        """Plots the original and preprocessed image"""


# ------------------------------------------------------------------------------------------------ #
#                               VISUALIZE DENOISER RESULTS                                         #
# ------------------------------------------------------------------------------------------------ #
class VisualizeDenoiserResults(VisualizeResults):
    """Visualizes Results of the Denoiser Experiment.

    Args:
        results_filepath (str): Path to file containing the results
        source_image_dir (str): Location of source images
        test_image_dir (str): Location of test images. Defaults to source_image_dir.
    """

    def __init__(
        self, results_filepath: str, source_image_dir: str, test_image_dir: str = None
    ) -> None:
        self._results_filepath = results_filepath
        self._source_image_dir = source_image_dir
        self._test_image_dir = test_image_dir or source_image_dir
        self._results = pd.read_csv(results_filepath)

    def plot_results(self) -> None:
        """Plots the results from the experiment."""
        fig, axes = plt.subplots(ncols=1, nrows=4, figsize=(12, 9))
        sns.lineplot(data=self._results, x="test_no", y="mse", hue="method", ax=axes[0])
        sns.lineplot(data=self._results, x="test_no", y="psnr", hue="method", ax=axes[1])
        sns.lineplot(data=self._results, x="test_no", y="ssim", hue="method", ax=axes[2])

        axes[0].set_title("Denoisers by Mean Squared Error")
        axes[1].set_title("Denoisers by Peak Signal to Noise Ratio")
        axes[2].set_title("Denoisers by Structural Similarity")
        fig.suptitle("Denoising Results", fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_images(self, metric: str, cmap: str = "gray") -> None:
        """Plots the original and preprocessed image"""
        methods = self._results["method"].unique()
        nrows = len(methods)
        ncols = 2
        image_height = 4
        image_width = 6
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(image_width * ncols, image_height * nrows)
        )

        for idx, method in enumerate(methods):
            if metric == "mse":
                data = self._results[
                    (self._results["method"] == method)
                    & (
                        self._results[metric]
                        == min(self._results[self._results["method"] == method][metric])
                    )
                ]
            else:
                data = self._results[
                    (self._results["method"] == method)
                    & (
                        self._results[metric]
                        == max(self._results[self._results["method"] == method][metric])
                    )
                ]

            source_filepath = os.path.join(
                self._source_image_dir, data["source_image_filepath"].values[0]
            )
            test_filepath = os.path.join(
                self._test_image_dir, data["test_image_filepath"].values[0]
            )
            source_image = cv2.imread(source_filepath)
            test_image = cv2.imread(test_filepath)
            axes[idx, 0].imshow(source_image, cmap=cmap)
            axes[idx, 0].set_title("Original Image")
            axes[idx, 1].imshow(test_image, cmap=cmap)
            title = f"{string.capwords(method.replace('_',' '))} Denoise Image\nParams: {data['params'].values[0]}"
            axes[idx, 1].set_title(title)

        title = f"Best Denoiser by {string.capwords(metric)}."
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.show()
