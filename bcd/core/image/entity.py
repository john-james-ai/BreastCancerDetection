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
# Modified   : Sunday October 29th 2023 02:25:15 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Module"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bcd.core.base import DataClass


# ------------------------------------------------------------------------------------------------ #
#                                         IMAGE                                                    #
# ------------------------------------------------------------------------------------------------ #
@dataclass(eq=False)
class Image(DataClass):
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
    created: datetime

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
            and int(self.aspect_ratio) == int(other.aspect_ratio)
            and self.min_pixel_value == other.min_pixel_value
            and self.max_pixel_value == other.max_pixel_value
            and self.range_pixel_values == other.range_pixel_values
            and int(self.mean_pixel_value) == int(other.mean_pixel_value)
            and self.median_pixel_value == other.median_pixel_value
            and int(self.std_pixel_value) == int(other.std_pixel_value)
            and self.filepath == other.filepath
            and self.fileset == other.fileset
            and self.cancer == other.cancer
            and self.transformer == other.transformer
            and self.task_id == other.task_id
        )

    def visualize(
        self, cmap: str = "jet", ax: plt.Axes = None, figsize: tuple = (8, 8)
    ) -> None:  # pragma: no cover
        """Plots the image on an axis

        Args:
            cmap (str): The colormap used to render the plot
            ax (plt.Axes): Matplotlib Axes object. Optional.
            figsize (tuple): Size of the image if a plt.Axes object is not provided.
                Default = (8,8)
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.pixel_data, cmap=cmap)
        ax.set_title(self.case_id)
        plt.show()

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
