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
# Modified   : Thursday October 26th 2023 12:04:06 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Module"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bcd.core.base import DataClass


# ------------------------------------------------------------------------------------------------ #
#                                         IMAGE                                                    #
# ------------------------------------------------------------------------------------------------ #
@dataclass(eq=False)
class Image(DataClass):
    id: str
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
    preprocessor: str
    task_id: str
    created: datetime

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
            "id": self.id,
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
            "preprocessor": self.preprocessor,
            "task_id": self.task_id,
            "created": self.created,
        }
        return pd.DataFrame(data=d, index=[0])
