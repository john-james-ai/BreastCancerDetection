#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/image/entity.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 10:27:45 am                                              #
# Modified   : Monday October 30th 2023 07:25:24 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Module"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bcd.core.base import Entity

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")
sns.set_palette(palette="Blues_r")

warnings.simplefilter(action="ignore", category=FutureWarning)


# ------------------------------------------------------------------------------------------------ #
#                                         IMAGE                                                    #
# ------------------------------------------------------------------------------------------------ #
@dataclass(eq=False)
class Image(Entity):
    """Image Object"""

    uid: str
    case_id: str
    mode: str
    stage_id: int
    stage: str
    left_or_right_breast: str
    image_view: str
    abnormality_type: str
    assessment: int
    breast_density: int
    bit_depth: int
    pixel_data: np.ndarray
    height: int
    width: int
    size: int
    aspect_ratio: float
    min_pixel_value: int
    max_pixel_value: int
    range_pixel_values: int
    mean_pixel_value: float
    median_pixel_value: int
    std_pixel_value: float
    filepath: str
    fileset: str
    cancer: bool
    transformer: str
    task_id: str
    created: datetime = field(compare=False)

    def __eq__(self, other: Image) -> bool:
        return (
            self.uid == other.uid
            and self.case_id == other.case_id
            and self.mode == other.mode
            and self.stage_id == other.stage_id
            and self.stage == other.stage
            and self.left_or_right_breast == other.left_or_right_breast
            and self.image_view == other.image_view
            and self.abnormality_type == other.abnormality_type
            and self.assessment == other.assessment
            and self.breast_density == other.breast_density
            and self.bit_depth == other.bit_depth
            and (self.pixel_data == other.pixel_data).all()
            and self.height == other.height
            and self.width == other.width
            and self.size == other.size
            and round(self.aspect_ratio, 1) == round(other.aspect_ratio, 1)
            and self.min_pixel_value == other.min_pixel_value
            and self.max_pixel_value == other.max_pixel_value
            and self.range_pixel_values == other.range_pixel_values
            and round(self.mean_pixel_value, 0) == round(other.mean_pixel_value, 0)
            and self.median_pixel_value == other.median_pixel_value
            and round(self.std_pixel_value, 1) == round(other.std_pixel_value, 1)
            and self.filepath == other.filepath
            and self.fileset == other.fileset
            and self.cancer == other.cancer
            and self.transformer == other.transformer
            and self.task_id == other.task_id
        )

    def difference(self, other: Image) -> dict:
        dict1 = self.as_dict()
        dict2 = other.as_dict()
        set1 = set(dict1.items())
        set2 = set(dict2.items())
        return set1 ^ set2

    def visualize(
        self,
        cmap: str = "gray",
        ax: plt.Axes = None,
        figsize: tuple = (8, 8),
        actual_size: bool = True,
    ) -> None:  # pragma: no cover
        """Plots the image on an axis

        Args:
            cmap (str): The colormap used to render the plot
            ax (plt.Axes): Matplotlib Axes object. Optional.
            figsize (tuple): Size of the image if a plt.Axes object is not provided.
                Default = (8,8)
            actual_size (bool): If True, the image is rendered at actual size
        """
        if actual_size:
            self._visualize_actual_size(cmap)
        else:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            ax.imshow(self.pixel_data, cmap=cmap)
            ax.set_title(self.case_id)
            plt.show()

    def _visualize_actual_size(self, cmap: str) -> None:  # pragma: no cover
        """Renders image at actual size"""
        dpi = 80
        height, width = self.pixel_data.shape

        # Computes the figure size to render the plot at actual size.
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide spines, ticks, etc.
        ax.axis("off")

        # Display the image.
        ax.imshow(self.pixel_data, cmap=cmap)

        plt.show()

    def histogram(self, ax: plt.Axes = None, figsize: tuple = (8, 8)) -> None:
        """Plots a histogram of image pixel values.

        Args:
            ax (plt.Axes): Matplotlib Axes object. Optional.
            figsize (tuple): Size of the image if a plt.Axes object is not provided.
                Default = (8,8)

        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        _ = ax.set_xlabel("Pixel Values")
        _ = sns.histplot(data=self.pixel_data.flatten(), ax=ax)

    def as_df(self) -> pd.DataFrame:
        d = {
            "uid": self.uid,
            "case_id": self.case_id,
            "mode": self.mode,
            "stage_id": self.stage_id,
            "stage": self.stage,
            "left_or_right_breast": self.left_or_right_breast,
            "image_view": self.image_view,
            "abnormality_type": self.abnormality_type,
            "assessment": self.assessment,
            "breast_density": self.breast_density,
            "bit_depth": self.bit_depth,
            "height": self.height,
            "width": self.width,
            "size": self.size,
            "aspect_ratio": self.aspect_ratio,
            "min_pixel_value": self.min_pixel_value,
            "max_pixel_value": self.max_pixel_value,
            "range_pixel_values": self.range_pixel_values,
            "mean_pixel_value": self.mean_pixel_value,
            "median_pixel_value": self.median_pixel_value,
            "std_pixel_value": self.std_pixel_value,
            "filepath": self.filepath,
            "fileset": self.fileset,
            "cancer": self.cancer,
            "transformer": self.transformer,
            "task_id": self.task_id,
            "created": self.created,
        }
        return pd.DataFrame(data=d, index=[0])
