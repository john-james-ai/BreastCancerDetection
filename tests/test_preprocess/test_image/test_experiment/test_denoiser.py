#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_preprocess/test_image/test_experiment/test_denoiser.py                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday November 3rd 2023 01:29:33 pm                                                #
# Modified   : Saturday November 4th 2023 06:41:04 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long, no-name-in-module, no-member
# ------------------------------------------------------------------------------------------------ #
import inspect
import json
import logging
import os
import shutil
from datetime import datetime
from unittest.mock import patch
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import pytest
from skimage.restoration import denoise_tv_chambolle

from bcd.preprocess.image.experiment.denoise import TotalVariationDenoisier

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"
# ------------------------------------------------------------------------------------------------ #
CLASS_LOCATION = "bcd.preprocess.image.experiment.denoise.TotalVariationDenoisier"
PARAMS_TEST = {"weight": np.arange(0.05, 0.3, 0.2)}
BEST_PARAMS = {"weight": 0.1}
DENOISER = denoise_tv_chambolle
SOURCE = "tests/data/2_exp"
DESTINATION = "tests/data/3_denoise"
UID = "5049df43-e4e8-4488-9b6f-9bd85de0f856"


RESULTS_FILEPATH = "tests/data/3_denoise/results.csv"
TEST_UUID = str(uuid4())
RESULTS = {
    "test_no": 1,
    "source_image_uid": UID,
    "source_image_filepath": UID + ".png",
    "test_image_uid": TEST_UUID,
    "test_image_filepath": TEST_UUID + ".png",
    "mode": "test",
    "stage_id": 1,
    "stage": "Artifact Removal",
    "image_view": "CC",
    "abnormality_type": "mass",
    "assessment": 3,
    "cancer": False,
    "method": denoise_tv_chambolle.__name__,
    "params": json.dumps(BEST_PARAMS),
    "mse": 1.24,
    "psnr": 23.8,
    "ssim": 10.3,
    "evaluated": datetime.now(),
}
OUT_FILEPATH = os.path.join(DESTINATION, TEST_UUID + ".png")
IN_FILEPATH = os.path.join(SOURCE, UID + ".png")


@pytest.mark.denoise
class TestDenoiseCalibrator:  # pragma: no cover
    # ============================================================================================ #
    def test_mode(self, current_mode):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        if current_mode != "test":
            msg = "\nCHANGE MODE TO TEST BEFORE RUNNING PYTEST!\nExiting pytest!\n"
            logger.exception(msg)
            pytest.exit(msg)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_setup(self):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        shutil.rmtree(os.path.dirname(OUT_FILEPATH))
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_save_image(self):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        dn = TotalVariationDenoisier(
            denoiser=DENOISER,
            params=PARAMS_TEST,
            source=SOURCE,
            destination=DESTINATION,
            n_jobs=6,
        )
        image = cv2.imread(IN_FILEPATH)
        dn.save_image(filename=RESULTS["test_image_filepath"], image=image)

        assert len(os.listdir(os.path.dirname(OUT_FILEPATH))) > 0
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_save_results(self):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        results = []
        for i in range(5):
            RESULTS["test_no"] = i
            results.append(RESULTS)
        df = pd.DataFrame(data=results)

        dn = TotalVariationDenoisier(
            denoiser=DENOISER, params=PARAMS_TEST, source=SOURCE, destination=DESTINATION, n_jobs=6
        )

        dn.save_results(results=df)
        assert os.path.exists(RESULTS_FILEPATH)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    @patch.object(TotalVariationDenoisier, "denoise_image")
    def test_denoise(self, mock_denoise_image):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        os.remove(RESULTS_FILEPATH)
        mock_denoise_image.return_value = RESULTS

        dn = TotalVariationDenoisier(
            denoiser=DENOISER, source=SOURCE, destination=DESTINATION, n_jobs=6
        )
        dn.run()
        assert os.path.exists(RESULTS_FILEPATH)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
