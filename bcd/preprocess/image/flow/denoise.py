#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/denoise.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 31st 2023 04:45:05 am                                               #
# Modified   : Sunday November 5th 2023 10:10:26 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Denoiser Task Module"""
import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from bcd.config import Config
from bcd.dal.io.image_io import ImageIO
from bcd.image import Image, ImageFactory
from bcd.preprocess.image.evaluate import Evaluation
from bcd.preprocess.image.flow.basetask import Task
from bcd.preprocess.image.flow.decorator import counter, timer
from bcd.preprocess.image.method.basemethod import Param


# ------------------------------------------------------------------------------------------------ #
#                              DENOISER TASK PARAMS                                                #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class DenoiserTaskParams(Param):
    """Encapsulates the parameters for the Denoiser objects."""

    parallel: bool = False
    n_jobs: int = 6


# ------------------------------------------------------------------------------------------------ #
#                                DENOISER TASK                                                     #
# ------------------------------------------------------------------------------------------------ #
class DenoiserTask(Task):
    """Converts DICOM images to PNG Format

    Args:
        task_params (DenoiserTaskParams): Parameters that control the task behavior.
        config (Config): The application configuration class
        io (ImageIO): The class responsible for image io
    """

    def __init__(
        self,
        config: Config = Config,
        io: ImageIO = ImageIO,
    ) -> None:
        super().__init__()
        self._config = config
        self._io = io

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @timer
    def run(self) -> None:
        proceed = self._check_for_existing_images()
        if proceed:
            test_data = self._get_test_data()
            if self.task_params.parallel:
                self._process_images_parallel(test_data=test_data)
            else:
                self._process_images(test_data=test_data)
        else:
            msg = "Task aborted by user."
            self._logger.info(msg)

    def _get_test_data(self) -> pd.DataFrame:
        """Returns the metadata for all images in the current mode."""

        # Read input files for the current mode.
        test_data = self.uow.image_repo.get_meta()
        test_data.reset_index(inplace=True, names="test_no")
        return test_data

    def _process_images(self, test_data: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            test_data (pd.DataFrame): DataFrame containing image metadata.
        """
        for _, orig_image_metadata in tqdm(test_data.iterrows(), total=test_data.shape[0]):
            (xformed_image, evaluation) = self._process_image(
                orig_image_metadata=orig_image_metadata
            )
            self.uow.image_repo.add(image=xformed_image)
            self.uow.eval_repo.add(evaluation=evaluation)

    def _process_images_parallel(self, test_data: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            test_data (pd.DataFrame): DataFrame containing image metadata.
        """
        with joblib.Parallel(n_jobs=self.task_params.n_jobs) as parallel:
            results = parallel(
                joblib.delayed(self._process_image)(orig_image_metadata)
                for _, orig_image_metadata in tqdm(test_data.iterrows(), total=len(test_data))
            )
        xformed_images = [item[0] for item in results]
        evaluations = [item[1] for item in results]
        for xformed_image in xformed_images:
            self.uow.image_repo.add(image=xformed_image)
        for evaluation in evaluations:
            self.uow.eval_repo.add(evaluation=evaluation)

    def _process_image(self, orig_image_metadata: pd.Series) -> Image:
        # Read the pixel data from file
        channels = 2
        if self.method.name == "BilateralFilter":
            channels = 3
        orig_image = self.uow.image_repo.get(uid=orig_image_metadata["uid"], channels=channels)
        # Execute the transformation
        xformed_pixels = self._transform_image(pixels=orig_image.pixel_data)

        # Create an image object
        xformed_image = ImageFactory.create(
            case_id=orig_image_metadata["case_id"],
            stage_id=self.stage_id,
            pixel_data=xformed_pixels,
            method=self.method.name,
            task_id=self.uid,
        )
        # Evaluate the image
        ev = Evaluation.evaluate(
            test_data=orig_image_metadata,
            orig_image=orig_image,
            test_image=xformed_image,
            stage_id=self.stage_id,
            method=self.method,
            params=self.method_params,
            comp_time=self.stage.duration,
        )

        return (xformed_image, ev)

    @counter
    def _transform_image(self, pixels: np.ndarray) -> np.ndarray:
        """Performs the image transformation."""
        return self._method.execute(image=pixels, params=self.method_params)

    def _check_for_existing_images(self) -> bool:
        """Returns True if no existing images or approved to append"""
        proceed = True
        condition = lambda df: df["stage_id"] == self.method.stage.uid
        count = self.uow.image_repo.count(condition)
        if count > 0:
            msg = f"There are already {count} images at the {self.method.stage.name} stage. Do you want to append? [Y/N]"
            proceed = "y" in input(msg).lower()
        return proceed
