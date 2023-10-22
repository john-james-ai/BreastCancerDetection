#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_data/test_entity/test_image_entity.py                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 02:14:19 pm                                              #
# Modified   : Sunday October 22nd 2023 02:06:41 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import inspect
from datetime import datetime
import pytest
import logging
import math

import pandas as pd
import numpy as np

from bcd.manage_data.entity.image import Image
from bcd.manage_data.io.image import ImageIO

DICOM_FP = "data/meta/2_clean/dicom.csv"
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.image
class TestImage:  # pragma: no cover
    # ============================================================================================ #
    def test_create(self, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        df = pd.read_csv(DICOM_FP)
        df = df.loc[df["series_description"] == "full mammogram images"]
        df = df.sample(n=1)
        io = ImageIO()
        pixel_data = io.read(filepath=df["filepath"])
        pixel_data = (pixel_data / 256).astype("uint8")
        logger.debug(f"Max pixel value {np.max(pixel_data, axis=None)}")
        logger.debug(f"Pixel shape {pixel_data.shape}")

        image = Image.create(
            case_id=df["case_id"].values[0],
            stage_id=0,
            bit_depth=8,
            pixel_data=pixel_data,
            cancer=df["cancer"].values[0],
            fileset=df["fileset"].values[0],
            task="TestImage",
            taskrun_id="some_taskrun_id",
        )

        logger.debug(image)

        assert image.id != df["id"].values[0]
        assert image.case_id == df["case_id"].values[0]
        assert image.cancer == df["cancer"].values[0]
        assert image.bit_depth == 8
        assert image.height == df["height"].values[0]
        assert image.width == df["width"].values[0]
        assert image.size == df["size"].values[0]
        assert math.isclose(image.aspect_ratio, df["aspect_ratio"].values[0], rel_tol=1e-05)
        assert image.min_pixel_value == df["min_pixel_value"].values[0]
        assert image.max_pixel_value < df["max_pixel_value"].values[0]
        assert image.range_pixel_values < df["range_pixel_values"].values[0]
        assert image.mean_pixel_value < df["mean_pixel_value"].values[0]
        # assert image.median_pixel_value < df["median_pixel_value"].values[0] May or may not be the same
        assert image.std_pixel_value != df["std_pixel_value"].values[0]
        assert image.filepath != df["filepath"].values[0]
        assert image.fileset == df["fileset"].values[0]
        assert image.mode == "test"
        assert image.stage_id == 0
        assert image.task == "TestImage"
        assert image.taskrun_id == "some_taskrun_id"

        io.write(pixel_data=pixel_data, filepath=image.filepath)
        assert os.path.exists(image.filepath)

        image2 = image.as_df()
        image2 = Image.from_df(df=image2)

        logger.debug(image2)

        assert image == image2

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_exception(self, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        df = pd.read_csv(DICOM_FP)
        df = df.loc[df["series_description"] == "full mammogram images"]
        df = df.sample(n=1)
        io = ImageIO()
        pixel_data = io.read(filepath=df["filepath"])
        pixel_data = (pixel_data / 256).astype("uint8")
        logger.debug(f"Max pixel value {np.max(pixel_data, axis=None)}")
        logger.debug(f"Pixel shape {pixel_data.shape}")

        with pytest.raises(ValueError):
            _ = Image.create(
                case_id=df["case_id"].values[0],
                stage_id=66,
                bit_depth=8,
                pixel_data=pixel_data,
                cancer=df["cancer"].values[0],
                fileset=df["fileset"].values[0],
                task="TestImage",
                taskrun_id="some_taskrun_id",
            )

        with pytest.raises(ValueError):
            _ = Image.create(
                case_id=df["case_id"].values[0],
                stage_id=0,
                bit_depth=18,
                pixel_data=pixel_data,
                cancer=df["cancer"].values[0],
                fileset=df["fileset"].values[0],
                task="TestImage",
                taskrun_id="some_taskrun_id",
            )
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
