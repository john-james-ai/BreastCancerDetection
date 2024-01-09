#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/explore/image/eda_image.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday December 19th 2023 03:49:16 pm                                              #
# Modified   : Monday January 8th 2024 08:05:41 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
import os
from typing import Callable, Union

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

from bcd.dal.image import ImageIO
from bcd.utils.image import grayscale
from bcd.utils.math import find_factors

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")
# pylint: disable=no-member


# ------------------------------------------------------------------------------------------------ #
class ImageExplorer:
    """Encapsulates Image Exploratory Data Analysis"""

    def __init__(self, filepath: str, io: ImageIO = ImageIO) -> None:
        self._filepath = os.path.abspath(filepath)
        self._meta = pd.read_csv(self._filepath)
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._io = io
        self._average_mean_pixel_value = None
        self._average_mean_pixel_value_std = None

    def summary(self) -> Union[plt.Figure, pd.DataFrame]:
        """Provides a summary of counts by fileset, label and abnormality type."""
        # Summarizing number of images by pathology and train vs test set
        features = ["fileset", "cancer"]
        data = (
            self._meta.groupby(by=features)["mmg_id"]
            .count()
            .sort_index(level=0, ascending=False)
            .reset_index()
        )
        labels = ["Dataset", "Cancer", "Count"]
        data.columns = labels

        # Obtain total number of images for the title
        total = data["Count"].sum()

        # Using the barplot API as it is more flexible.
        fig, ax = plt.subplots(figsize=(12, 4))
        ax = sns.barplot(data=data, x="Dataset", y="Count", hue="Cancer", ax=ax)
        _ = ax.set_title(f"CBIS-DDSM Dataset Summary\nFull Mammogram Count: {total}")

        # Add Counts to Plot
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate(
                text=f"{int(y)}",
                xy=(x.mean(), y),
                ha="center",
                va="bottom",
            )
        # Summarize Stats in Dataframe
        data = data.pivot(index="Dataset", columns="Cancer", values="Count").sort_index(
            ascending=False
        )
        data.columns = ["Benign", "Malignant"]
        data["Total"] = data.sum(axis=1, numeric_only=True)
        data["Split"] = data["Total"] / data["Total"].sum()
        data["% Benign"] = round(data["Benign"] / data["Total"] * 100, 0).astype("int")
        data["% Malignant"] = round(data["Malignant"] / data["Total"] * 100, 0).astype(
            "int"
        )
        data = data[
            ["Benign", "% Benign", "Malignant", "% Malignant", "Total", "Split"]
        ]

        return fig, data

    def get_data(self, x: str, condition: Callable = None) -> np.ndarray:
        """Returns a quantitative variable optionally conditioned

        Args:
            x (str): The variable to return
            condition (Callable): Optional condition to apply
        """
        df = self._meta
        if condition:
            df = self._meta[condition]
        return df[x].to_numpy()

    def test_centrality(
        self, a: np.ndarray, b: np.ndarray, alternative: str = "less"
    ) -> float:
        """Tests centrality of x by y

        Args:
            x (str): The variable to be tested
            y (str): The variable defining the groups
        """
        t, pvalue = ttest_ind(a=a, b=b, equal_var=False, alternative=alternative)
        return t, pvalue

    def analyze_resolution(self) -> Union[plt.Figure, pd.DataFrame]:
        """Plots rows vs cols, and aspect ratio"""

        features = ["rows", "cols", "aspect_ratio"]
        stats = self._meta[features].describe().T

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        # Height Histogram
        axes[0] = sns.histplot(data=self._meta, x="rows", element="poly", ax=axes[0])
        axes[0].set_title("Distribution of Image Height")

        # Width Histogram
        axes[1] = sns.histplot(data=self._meta, x="cols", element="poly", ax=axes[1])
        axes[1].set_title("Distribution of Image Width")

        # Width Histogram
        axes[2] = sns.histplot(
            data=self._meta, x="aspect_ratio", element="poly", ax=axes[2]
        )
        axes[2].set_title("Distribution of Image Aspect Ratio")

        fig.suptitle("Resolution Analysis", fontsize=12)

        plt.tight_layout()

        return fig, stats

    def analyze_pixel_values(
        self, by: str = "cancer"
    ) -> Union[plt.Figure, pd.DataFrame]:
        """Plots mean and standard deviation of pixel values and returns statistics."""
        fig = plt.figure(tight_layout=True, figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2)

        # Plot Mean and Standard Deviation
        ax0 = fig.add_subplot(gs[0, :])
        ax0 = sns.scatterplot(
            data=self._meta,
            x="mean_pixel_value",
            y="std_pixel_value",
            hue=by,
            style=by,
            ax=ax0,
        )
        ax0.set_title("Mean and Standard Deviation of Pixel Intensities by Pathology")

        # Mean Histogram
        ax1 = fig.add_subplot(gs[1, 0])
        ax1 = sns.histplot(
            data=self._meta,
            x="mean_pixel_value",
            hue="cancer",
            ax=ax1,
            element="poly",
            stat="density",
        )
        ax1.set_title("Mean Pixel Intensities by Pathology")

        # Standard Deviation Histogram
        ax2 = fig.add_subplot(gs[1, 1])
        ax2 = sns.histplot(
            data=self._meta,
            x="std_pixel_value",
            hue="cancer",
            ax=ax2,
            element="poly",
            stat="density",
        )
        ax2.set_title("Standard Deviation Pixel Intensities by Pathology")

        fig.suptitle("Pixel Intensity Analysis", fontsize=12)

        # Compute statistics
        features = ["cancer", "mean_pixel_value", "std_pixel_value"]
        if by:
            stats = self._meta[features].groupby(by=by).describe()
        else:
            stats = self._meta[features].describe()

        return fig, stats

    def visualize(
        self,
        condition: Callable = None,
        n: int = 50,
        rows: int = 9,
        cols: int = 12,
        cmap: str = "turbo",
        sort_by: Union[str, list] = None,
        histogram: bool = False,
        title: str = None,
        random_state: int = None,
    ) -> plt.Figure:
        """Plots a grid of images

        Args:
            condition (Callable): Lambda expression used to select cases.
            n (int): Number of images to plot. Must be a multiple of nrows.
            rows (int): Height of image in inches.
            cols (int): Width of image in inches.
            cmap (str): Color map for matplotlib. Default = 'gray'.
            sort_by (str): Value to sort the images by.
            label (str): The label to use for each image.
            histogram (bool): Whether to plot the image histogram, instead of the image.
            title (str): Title of the plot.
            random_state (int): Seed for pseudo random sampling
        """

        df = self._meta
        if condition:
            df = self._meta[condition]

        # The number of plots must not be greater than the number of
        # cases.
        n = min(n, len(df))

        # Find nrows and ncols. Note, if n is prime,
        # nrows and ncols is approximated. Hence, n must be
        # reset accordingly and cases must be resampled for
        # the new value of n.
        nrows, ncols = find_factors(n, non_prime_approx=True)
        n = nrows * ncols
        df = df.sample(n=n, random_state=random_state)

        if sort_by:
            df = df.sort_values(by=sort_by)

        # Create figure and axes objects, then  iteratively plot images.
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(cols, rows))
        axes = axes.flatten()
        for ax, (_, row) in zip(axes, df.iterrows()):
            img = self._io.read(row["filepath"])
            img = grayscale(image=img)
            if histogram:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                ax.plot(hist)
            else:
                ax.imshow(img, cmap=cmap, aspect="auto")
            ax.set_title(f"{row['patient_id']}\n{row['pathology']}", fontsize=8)
            ax.axis("off")

        title = title or "CBIS-DDSM"
        fig.suptitle(title)
        plt.tight_layout()
        return fig
