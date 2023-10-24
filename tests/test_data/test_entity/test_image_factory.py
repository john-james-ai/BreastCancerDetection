#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_data/test_entity/test_image_factory.py                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 02:14:19 pm                                              #
# Modified   : Monday October 23rd 2023 02:56:55 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging
import math
import shutil

import pandas as pd

from bcd.manage_data.entity.image import Image
from bcd.manage_data.io.image import ImageIO

DICOM_FP = "data/meta/2_clean/dicom.csv"
CASE_ID = "Calcification-Test_P_01030_RIGHT_MLO_2"
TASK = "TestCaseFactory"
TASK2 = "TestCaseFactory2"
TASKRUN_ID = "some_taskrun_id"
TASKRUN_ID2 = "some_taskrun_id2"
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
        factory = container.repo.factory()
        df = pd.read_csv(DICOM_FP)
        df = df.loc[df["case_id"] == CASE_ID]
        io = ImageIO()
        pixel_data = io.read(df["filepath"])

        image = factory.create(
            case_id=CASE_ID,
            pixel_data=pixel_data,
            stage_id=0,
            task="TaskCreate",
            taskrun_id=TASKRUN_ID,
        )
        io = ImageIO()
        io.write(pixel_data=image.pixel_data, filepath=image.filepath)

        logger.debug(image)

        assert isinstance(image, Image)
        assert image.id != df["id"].values[0]
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
        assert image.stage_id == 0
        assert image.task == "TaskCreate"
        assert image.taskrun_id == TASKRUN_ID

        # Test From DF
        image2 = image.as_df()
        image2 = factory.from_df(df=image2)
        assert isinstance(image2, Image)

        logger.debug(image2)
        logger.debug(f"Shape image: {image.pixel_data.shape}")
        logger.debug(f"Shape image2: {image2.pixel_data.shape}")

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
    def test_exception(self, container, caplog):
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
        factory = container.repo.factory()
        df = pd.read_csv(DICOM_FP)
        df = df.loc[df["case_id"] == CASE_ID]
        io = ImageIO()
        pixel_data = io.read(df["filepath"])

        with pytest.raises(KeyError):
            _ = factory.create(
                case_id=CASE_ID,
                pixel_data=pixel_data,
                stage_id=99,
                task=TASK,
                taskrun_id=TASKRUN_ID,
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

    # ============================================================================================ #
    def test_teardown(self, caplog):
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
        FP = "tests/data/images"
        shutil.rmtree(FP, ignore_errors=True)

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
