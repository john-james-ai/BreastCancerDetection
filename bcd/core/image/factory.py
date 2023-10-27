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
# Modified   : Friday October 27th 2023 12:42:25 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Module"""
from __future__ import annotations
import os
from datetime import datetime
from uuid import uuid4
import logging

import pandas as pd
import numpy as np

from bcd.config import Config
from bcd.core.base import STAGES
from bcd.infrastructure.io.image import ImageIO
from bcd.core.image.entity import Image


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

        # Convert numpy datetime64 to python datetime.
        timestamp = ((df['created'].values[0] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1,'s'))
        created = datetime.utcfromtimestamp(timestamp)

        io = ImageIO()
        pixel_data = io.read(filepath=df["filepath"])
        return Image(
            uid=df["uid"].values[0],
            case_id=df["case_id"].values[0],
            mode=df["mode"].values[0],
            stage_uid=df["stage_uid"].values[0],
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
            preprocessor=df["preprocessor"].values[0],
            task_id=df["task_id"].values[0],
        )

    def create(
        self,
        case_id: str,
        stage_uid: int,
        pixel_data: np.ndarray,
        preprocessor: str,
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
            stage_uid (int): The preprocessing stage identifier
            pixel_data (np.ndarray): Pixel data in numpy array format.
            preprocessor (str): The name of the preprocessor
            task_id (str): The UUID for the specific task.

        Returns
            Image Object.

        """
        uid = str(uuid4())

        stage = self._get_stage(uid=stage_uid)

        case = self._cases.loc[self._cases["case_id"] == case_id]

        mode = self._config.mode

        basedir = self._config.image_directory

        io = ImageIO()
        filepath = io.get_filepath(uid=uid, basedir=basedir, format="png")

        return Image(
            uid=uid,
            case_id=case_id,
            mode=mode,
            stage_uid=stage_uid,
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

    def _get_stage(self, uid: int) -> None:
        try:
            return STAGES[uid]
        except KeyError:
            msg = f"Stage identifier = {uid} is invalid. Valid values are: {STAGES.keys()}"
            self._logger.exception(msg)
            raise ValueError(msg)
