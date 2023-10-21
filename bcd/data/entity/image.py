#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/entity/image.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 10:27:45 am                                              #
# Modified   : Saturday October 21st 2023 01:25:21 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Module"""
from __future__ import annotations
from datetime import datetime
from uuid import uuid4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..structure.dataclass import DataClass
from ..io.image import ImageIO


# ------------------------------------------------------------------------------------------------ #
STAGES = {
    0: "original",
    1: "denoise",
    2: "enhance",
    3: "artifact_removal",
    4: "pectoral_removal",
    5: "reshape",
}
# ------------------------------------------------------------------------------------------------ #


class Image(DataClass):
    id: str
    case_id: str
    state: str
    stage_id: int
    stage: str
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
    format: str
    filepath: str
    created: datetime
    taskrun_id: str

    def load_image(self) -> np.ndarray:
        io = ImageIO()
        self.pixel_data = io.read(filepath=self.filepath)

    @staticmethod
    def save_image(self) -> None:
        io = ImageIO()
        io.write(filepath=self.filepath, pixel_data=self.pixel_data)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Image:
        io = ImageIO()
        pixel_data = io.read(filepath=df["filepath"])
        return cls(
            id=df["id"],
            case_id=df["case_id"],
            state=df["state"],
            stage_id=df["stage_id"],
            stage=df["stage"],
            bit_depth=df["bit_depth"],
            pixel_data=pixel_data,
            height=df["height"],
            width=df["width"],
            size=df["size"],
            aspect_ratio=df["aspect_ratio"],
            min_pixel_value=df["min_pixel_value"],
            max_pixel_value=df["max_pixel_value"],
            range_pixel_values=df["range_pixel_values"],
            mean_pixel_value=df["mean_pixel_value"],
            median_pixel_value=df["median_pixel_value"],
            std_pixel_value=df["std_pixel_value"],
            format=df["format"],
            filepath=df["filepath"],
            created=df["created"],
            taskrun_id=df["taskrun_id"],
        )

    @classmethod
    def create(
        cls,
        case_id: str,
        state: str,
        stage_id: int,
        bit_depth: int,
        pixel_data: np.ndarray,
        format: str,
        taskrun_id: str,
    ) -> Image:
        io = ImageIO()
        filepath = io.get_filepath(state=state, case_id=case_id, format=format)
        return cls(
            id=str(uuid4()),
            case_id=case_id,
            state=state,
            stage_id=stage_id,
            stage=STAGES[stage_id],
            bit_depth=bit_depth,
            pixel_data=pixel_data,
            height=pixel_data.shape[0],
            width=pixel_data.shape[1],
            size=pixel_data.size,
            aspect_ratio=pixel_data.shape[1] / pixel_data.shape[0],
            min_pixel_value=pixel_data.min(axis=None),
            max_pixel_value=pixel_data.max(axis=None),
            range_pixel_values=pixel_data.max(axis=None) - pixel_data.min(axis=None),
            mean_pixel_value=pixel_data.mean(axis=None),
            median_pixel_value=pixel_data.median(axis=None),
            std_pixel_value=pixel_data.std(axis=None),
            format=format,
            filepath=filepath,
            created=datetime.now(),
            taskrun_id=taskrun_id,
        )

    def visualize(self, cmap: str = "jet", ax: plt.Axes = None, figsize: tuple = (8, 8)) -> None:
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
