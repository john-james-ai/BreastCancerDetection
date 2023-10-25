#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/manage_data/entity/image.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 10:27:45 am                                              #
# Modified   : Wednesday October 25th 2023 05:18:42 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Module"""
from __future__ import annotations
from dataclasses import dataclass
import os
from datetime import datetime
from uuid import uuid4
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bcd.config import Config
from bcd.manage_data import STAGES
from bcd.manage_data.io.image import ImageIO
from bcd.manage_data.entity.base import Entity


# ------------------------------------------------------------------------------------------------ #
#                                         IMAGE                                                    #
# ------------------------------------------------------------------------------------------------ #
@dataclass(eq=False)
class Image(Entity):
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


# ------------------------------------------------------------------------------------------------ #
#                                    IMAGE FACTORY                                                 #
# ------------------------------------------------------------------------------------------------ #
class ImageFactory:
    """Creates Image Objects"""

    def __init__(self, case_fp: str, config: Config) -> None:
        case_fp = os.path.abspath(case_fp)
        self._cases = pd.read_csv(case_fp)
        self._cases = self._cases.loc[self._cases["series_description"] == "full mammogram images"]
        self._config = config()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def from_df(self, df: pd.DataFrame) -> Image:
        """Creates an image from a DataFrame

        This method is called to reconstitute an image from the database.

        Args:
            df (pd.DataFrame): Dataframe containing image metadata.

        Returns:
            Image object
        """
        io = ImageIO()
        pixel_data = io.read(filepath=df["filepath"])
        return Image(
            id=df["id"].values[0],
            case_id=df["case_id"].values[0],
            mode=df["mode"].values[0],
            stage_id=df["stage_id"].values[0],
            stage=df["stage"].values[0],
            left_or_right_breast=df["left_or_right_breast"].values[0],
            image_view=df["image_view"].values[0],
            abnormality_type=df["abnormality_type"].values[0],
            assessment=df["assessment"].values[0],
            breast_density=df["breast_density"].values[0],
            bit_depth=df["bit_depth"].values[0],
            pixel_data=pixel_data,
            height=df["height"].values[0],
            width=df["width"].values[0],
            size=df["size"].values[0],
            aspect_ratio=df["aspect_ratio"].values[0],
            min_pixel_value=df["min_pixel_value"].values[0],
            max_pixel_value=df["max_pixel_value"].values[0],
            range_pixel_values=df["range_pixel_values"].values[0],
            mean_pixel_value=df["mean_pixel_value"].values[0],
            median_pixel_value=df["median_pixel_value"].values[0],
            std_pixel_value=df["std_pixel_value"].values[0],
            filepath=df["filepath"].values[0],
            fileset=df["fileset"].values[0],
            cancer=df["cancer"].values[0],
            created=df["created"].values[0],
            preprocessor=df["preprocessor"].values[0],
            task_id=df["task_id"].values[0],
        )

    def create(
        self,
        case_id: str,
        stage_id: int,
        pixel_data: np.ndarray,
        preprocessor: str,
        task_id: str,
    ) -> Image:
        """Creates an image from pizel data.

        Args:
            case_id (str): Unique identifier for a case.
            stage_id (int): The preprocessing stage identifier
            pixel_data (np.ndarray): Pixel data in numpy array format.
            preprocessor (str): The name of the preprocessor
            task_id (str): The UUID for the specific task.

        Returns
            Image Object.

        """
        id = str(uuid4())

        stage = self._get_stage(stage_id=stage_id)

        case = self._cases.loc[self._cases["case_id"] == case_id]

        mode = self._config.get_mode()

        basedir = self._config.get_image_directory()

        io = ImageIO()
        filepath = io.get_filepath(id=id, basedir=basedir, format="png")
        io.write(pixel_data=pixel_data, filepath=filepath)

        return Image(
            id=id,
            case_id=case_id,
            mode=mode,
            stage_id=stage_id,
            stage=stage,
            left_or_right_breast=case["left_or_right_breast"].values[0],
            image_view=case["image_view"].values[0],
            abnormality_type=case["abnormality_type"].values[0],
            assessment=case["assessment"].values[0],
            breast_density=case["breast_density"].values[0],
            bit_depth=case["bit_depth"].values[0],
            pixel_data=pixel_data,
            height=pixel_data.shape[0],
            width=pixel_data.shape[1],
            size=pixel_data.size,
            aspect_ratio=pixel_data.shape[1] / pixel_data.shape[0],
            min_pixel_value=np.min(pixel_data, axis=None),
            max_pixel_value=np.max(pixel_data, axis=None),
            range_pixel_values=pixel_data.max(axis=None) - pixel_data.min(axis=None),
            mean_pixel_value=pixel_data.mean(axis=None),
            median_pixel_value=np.median(pixel_data, axis=None).astype(np.uint8),
            std_pixel_value=np.std(pixel_data, axis=None),
            filepath=filepath,
            fileset=case["fileset"].values[0],
            cancer=case["cancer"].values[0],
            created=datetime.now(),
            preprocessor=preprocessor,
            task_id=task_id,
        )

    def _get_stage(self, stage_id: int) -> None:
        try:
            return STAGES[stage_id]
        except KeyError:
            msg = f"Stage identifier = {stage_id} is invalid. Valid values are: {STAGES.keys()}"
            self._logger.exception(msg)
            raise ValueError(msg)
