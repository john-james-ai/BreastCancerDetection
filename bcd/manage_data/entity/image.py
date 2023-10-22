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
# Modified   : Sunday October 22nd 2023 02:06:05 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Module"""
from __future__ import annotations
from dataclasses import dataclass
import os
import sys
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bcd.manage_data import STAGES
from bcd.manage_data.io.image import ImageIO
from bcd.manage_data.structure.dataclass import DataClass

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #


@dataclass(eq=False)
class Image(DataClass):
    id: str
    case_id: str
    mode: str
    stage_id: int
    stage: str
    cancer: bool
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
    created: datetime
    task: str
    taskrun_id: str

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Image:
        io = ImageIO()
        pixel_data = io.read(filepath=df["filepath"])
        return cls(
            id=df["id"].values[0],
            case_id=df["case_id"].values[0],
            mode=df["mode"].values[0],
            stage_id=df["stage_id"].values[0],
            stage=df["stage"].values[0],
            cancer=df["cancer"].values[0],
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
            median_pixel_value=df["median_pixel_value"].values[0].astype(np.uint8),
            std_pixel_value=df["std_pixel_value"].values[0],
            filepath=df["filepath"].values[0],
            fileset=df["fileset"].values[0],
            created=df["created"].values[0],
            task=df["task"].values[0],
            taskrun_id=df["taskrun_id"].values[0],
        )

    @classmethod
    def create(
        cls,
        case_id: str,
        stage_id: int,
        bit_depth: int,
        pixel_data: np.ndarray,
        cancer: bool,
        fileset: str,
        task: str,
        taskrun_id: str,
    ) -> Image:
        if stage_id not in STAGES.keys():
            msg = f"Invalid stage_id. Valid values: {STAGES.keys()}"
            logger.exception(msg)
            raise ValueError(msg)

        if bit_depth not in [8, 16]:
            msg = f"Bit bepth {bit_depth} is invalid. Valid values are 8 and 16."
            logger.exception(msg)
            raise ValueError(msg)

        id = str(uuid4())

        mode = os.getenv("MODE")

        io = ImageIO()
        filepath = io.get_filepath(id=id, mode=mode, format="png")

        return cls(
            id=id,
            case_id=case_id,
            mode=mode,
            stage_id=stage_id,
            stage=STAGES[stage_id],
            cancer=cancer,
            bit_depth=bit_depth,
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
            fileset=fileset,
            created=datetime.now(),
            task=task,
            taskrun_id=taskrun_id,
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
