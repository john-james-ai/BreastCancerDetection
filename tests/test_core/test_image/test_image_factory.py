#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_core/test_image/test_image_factory.py                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 02:14:19 pm                                              #
# Modified   : Saturday October 28th 2023 03:39:53 pm                                              #
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

from bcd.core.image.entity import Image
from bcd.dal.io.image import ImageIO

DICOM_FP = "data/meta/2_clean/dicom.csv"
CASE_ID = "Calcification-Test_P_01030_RIGHT_MLO_2"
PREPROCESSOR = "TestCaseFactory"
PREPROCESSOR2 = "TestCaseFactory2"
TASK_ID = "some_task_id"
TASK_ID2 = "some_task_id2"
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.image_factory
class TestImageFactory:  # pragma: no cover
    # ============================================================================================ #
    def test_creation(self, container, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        factory = container.repo.factory()
        df = pd.read_csv(DICOM_FP)
        df = df.loc[df["case_id"] == CASE_ID]
        io = ImageIO()
        pixel_data = io.read(df["filepath"])

        image = factory.create(
            case_id=CASE_ID,
            stage_uid=0,
            pixel_data=pixel_data,
            preprocessor=PREPROCESSOR,
            task_id=TASK_ID,
        )
        logger.debug(image)

        assert isinstance(image, Image)
        assert image.uid != df["uid"].values[0]
        assert image.case_id == df["case_id"].values[0]
        assert image.cancer == df["cancer"].values[0]
        assert image.bit_depth == df["bit_depth"].values[0]
        assert image.height == df["height"].values[0]
        assert image.width == df["width"].values[0]
        assert image.size == df["size"].values[0]
        assert math.isclose(image.aspect_ratio, df["aspect_ratio"].values[0], rel_tol=1e-05)
        assert image.min_pixel_value == df["min_pixel_value"].values[0]
        assert image.max_pixel_value == df["max_pixel_value"].values[0]
        assert image.range_pixel_values == df["range_pixel_values"].values[0]
        assert image.mean_pixel_value == df["mean_pixel_value"].values[0]
        assert image.median_pixel_value == df["median_pixel_value"].values[0]
        assert math.isclose(image.std_pixel_value, df["std_pixel_value"].values[0], rel_tol=1e-5)
        assert image.filepath != df["filepath"].values[0]
        assert image.fileset == df["fileset"].values[0]
        assert image.mode == "test"
        assert image.stage_uid == 0
        assert image.stage == "converted"
        assert image.preprocessor == PREPROCESSOR
        assert image.task_id == TASK_ID

        assert os.path.exists(image.filepath)

        # Test From DF
        image2 = image.as_df()
        image2 = factory.from_df(df=image2)
        assert isinstance(image2, Image)

        logger.debug(image2)
        logger.debug(f"Shape image: {image.pixel_data.shape}")
        logger.debug(f"Shape image2: {image2.pixel_data.shape}")

        assert image == image2

        os.remove(image.filepath)

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
    def test_exception(self, container, caplog):
        start = datetime.now()
        logger.info(f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        factory = container.repo.factory()
        df = pd.read_csv(DICOM_FP)
        df = df.loc[df["case_id"] == CASE_ID]
        io = ImageIO()
        pixel_data = io.read(df["filepath"])

        with pytest.raises(ValueError):
            _ = factory.create(
                case_id=CASE_ID,
                pixel_data=pixel_data,
                stage_uid=99,
                preprocessor=PREPROCESSOR,
                task_id=TASK_ID,
            )

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}")
        logger.info(single_line)
