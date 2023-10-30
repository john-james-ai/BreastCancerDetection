#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/image/factory.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 10:55:08 pm                                             #
# Modified   : Monday October 30th 2023 06:50:27 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Module"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bcd.core.base import Stage
from bcd.core.image.entity import Image
from bcd.dal.io.image import ImageIO
from bcd.utils.date import to_datetime

load_dotenv()


# ------------------------------------------------------------------------------------------------ #
#                                    IMAGE FACTORY                                                 #
# ------------------------------------------------------------------------------------------------ #
class ImageFactory:
    """Creates Image Objects"""

    def __init__(self, metadata_filepath: str, mode: str, directory: str, io: ImageIO) -> None:
        metadata_filepath = os.path.abspath(metadata_filepath)
        image_metadata = pd.read_csv(metadata_filepath)
        self._image_metadata = image_metadata.loc[
            image_metadata["series_description"] == "full mammogram images"
        ]
        self._mode = mode
        self._directory = directory
        self._io = io
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def from_df(self, df: pd.DataFrame) -> Image:
        """Creates an image from a DataFrame

        This method is called to reconstitute an image from the database.

        Args:
            df (pd.DataFrame): Dataframe containing image metadata.

        Returns:
            Image object
        """

        # Convert numpy datetime64 to python datetime.
        created = to_datetime(dt=df["created"].values[0])

        pixel_data = self._io.read(filepath=df["filepath"])
        return Image(
            uid=df["uid"].values[0],
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
            created=created,
            transformer=df["transformer"].values[0],
            task_id=df["task_id"].values[0],
        )

    def create(
        self,
        case_id: str,
        stage_id: int,
        pixel_data: np.ndarray,
        transformer: str,
        task_id: str,
    ) -> Image:
        """Creates an image from pizel data.

        Note: It does not save the image to disk. Persistence
        functions are the domain of the repository. Though, this method
        can read the pixel data from file, in order to reconstitute a
        fully formed Image object, persistence of the image and
        the metadata are controlled by the repository.

        Args:
            case_id (str): Unique identifier for a case.
            stage_id (int): The preprocessing stage identifier
            pixel_data (np.ndarray): Pixel data in numpy array format.
            transformer (str): The name of the transformer
            task_id (str): The UUID for the specific task.

        Returns
            Image Object.

        """
        uid = str(uuid4())

        stage = Stage(uid=stage_id).name

        case = self._image_metadata.loc[self._image_metadata["case_id"] == case_id]

        filepath = self._io.get_filepath(uid=uid, basedir=self._directory, fileformat="png")

        return Image(
            uid=uid,
            case_id=case_id,
            mode=self._mode,
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
            transformer=transformer,
            task_id=task_id,
        )
