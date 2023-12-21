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
# Modified   : Thursday December 21st 2023 07:25:55 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
from typing import Callable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bcd.data.image import ImageIO
from bcd.utils.math import find_factors

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")
sns.set_palette("Blues_r")


# ------------------------------------------------------------------------------------------------ #
class ImageExplorer:
    """Encapsulates Image Exploratory Data Analysis"""

    __FP = "data/meta/2_clean/dicom.csv"

    def __init__(self, io: ImageIO = ImageIO) -> None:
        self._meta = pd.read_csv(self.__FP)
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._io = io

    def summary(self) -> pd.DataFrame:
        """Provides a summary of counts by fileset, label and abnormality type."""
        # Summarizing number of images by pathology and train vs test set
        features = ["fileset", "cancer"]
        data = (
            self._meta.groupby(by=features)["uid"]
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

    def resolution_stats(self) -> pd.DataFrame:
        """Provides descriptive statistics for height, width, and aspect ratio."""
        features = ["height", "width", "aspect_ratio"]
        return self._meta[features].describe().T

    def pixel_stats(self, groupby: list = None) -> pd.DataFrame:
        """Summarizes pixel descriptive statistics."""
        features = [
            "median_pixel_value",
            "mean_pixel_value",
            "max_pixel_value",
            "std_pixel_value",
        ]
        if groupby:
            return self._meta[features].groupby(by=groupby).describe().T
        else:
            return self._meta[features].describe().T

    def plot_resolution_stats(self) -> plt.Figure:
        """Plots height vs width, and aspect ratio"""
        fig = plt.figure(tight_layout=True, figsize=(12, 12))
        gs = gridspec.GridSpec(4, 2)

        # Scatterplot
        ax0 = fig.add_subplot(gs[0, :])
        ax0 = sns.scatterplot(
            data=self._meta, x="width", y="height", hue="cancer", ax=ax0
        )
        ax0.set_title("Dimensions by Label")

        # Height Histogram
        ax1 = fig.add_subplot(gs[1, 0])
        ax1 = sns.histplot(
            data=self._meta,
            x="height",
            hue="cancer",
        )
        ax1.set_title("Height")

        # Width Histogram
        ax2 = fig.add_subplot(gs[1, 1])
        ax2 = sns.histplot(
            data=self._meta,
            x="width",
            hue="cancer",
        )
        ax2.set_title("Width")

        # Height Boxplot
        ax3 = fig.add_subplot(gs[2, 0])
        ax3 = sns.boxplot(
            data=self._meta,
            x="height",
            hue="cancer",
        )
        ax3.set_title("Height")

        # Width Boxplot
        ax3 = fig.add_subplot(gs[2, 1])
        ax3 = sns.boxplot(
            data=self._meta,
            x="width",
            hue="cancer",
        )
        ax3.set_title("Width")

        # Aspect Ratio Hist
        ax4 = fig.add_subplot(gs[3, :])
        ax4 = sns.histplot(
            data=self._meta,
            x="aspect_ratio",
            hue="cancer",
        )
        ax4.set_title("Aspect Ratio")

        fig.suptitle("Dimension Analysis", fontsize=12)

        plt.tight_layout()

        return fig

    def plot_pixel_stats(self) -> plt.Figure:
        """Plots mean and standard deviation of pixel values"""
        fig = plt.figure(tight_layout=True, figsize=(12, 12))
        gs = gridspec.GridSpec(3, 2)

        # Plot Mean and Standard Deviation
        ax0 = fig.add_subplot(gs[0, :])
        ax0 = sns.scatterplot(
            data=self._meta,
            x="mean_pixel_value",
            y="std_pixel_value",
            hue="cancer",
            ax=ax0,
        )
        ax0.set_title("Mean and Standard Deviation of Pixel Intensities")

        # Mean Histogram
        ax1 = fig.add_subplot(gs[1, 0])
        ax1 = sns.histplot(data=self._meta, x="mean_pixel_value", hue="cancer", ax=ax1)
        ax1.set_title("Mean Pixel Intensities")

        # Standard Deviation Histogram
        ax2 = fig.add_subplot(gs[1, 1])
        ax2 = sns.histplot(data=self._meta, x="std_pixel_value", hue="cancer", ax=ax2)
        ax2.set_title("Standard Deviation Pixel Intensities")

        # Mean Boxplot
        ax1 = fig.add_subplot(gs[2, 0])
        ax1 = sns.boxplot(data=self._meta, x="mean_pixel_value", hue="cancer", ax=ax1)
        ax1.set_title("Mean Pixel Values")

        # Standard Deviation Boxplot
        ax2 = fig.add_subplot(gs[2, 1])
        ax2 = sns.boxplot(data=self._meta, x="std_pixel_value", hue="cancer", ax=ax2)
        ax2.set_title("Standard Deviation Pixel Intensities")

        fig.suptitle("Pixel Intensity Analysis", fontsize=12)

        return fig

    def visualize(
        self,
        condition: Callable = None,
        n: int = 50,
        height: int = 9,
        width: int = 12,
        cmap: str = "gray",
    ) -> plt.Figure:
        """Plots a grid of images

        Args:
            condition (Callable): Lambda expression used to select cases.
            n (int): Number of images to plot. Must be a multiple of nrows.
            height (int): Height of image in inches.
            width (int): Width of image in inches.
            cmap (str): Color map for matplotlib. Default = 'gray'.
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
        df = df.sample(n=n)

        # Create figure and axes objects, then  iteratively plot images.
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
        axes = axes.flatten()
        for ax, (_, row) in zip(axes, df.iterrows()):
            img = self._io.read(row["filepath"])
            ax.imshow(img, cmap=cmap, aspect="auto")
            ax.set_title(f"{row['case_id']}\n{row['pathology']}", fontsize=4)
            ax.axis("off")

        fig.suptitle("CBIS-DDSM")
        plt.tight_layout()
        return fig
