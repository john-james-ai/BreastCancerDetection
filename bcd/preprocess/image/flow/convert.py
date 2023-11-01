#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/convert.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 31st 2023 04:45:05 am                                               #
# Modified   : Wednesday November 1st 2023 07:07:51 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Converter Task Module"""
import logging
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from bcd.config import Config
from bcd.dal.io.image import ImageIO
from bcd.preprocess.image.flow.basetask import Task
from bcd.preprocess.image.flow.decorator import counter, timer
from bcd.preprocess.image.image import ImageFactory
from bcd.preprocess.image.method.basemethod import Param


# ------------------------------------------------------------------------------------------------ #
#                                   CONVERTER TASK PARAMS                                          #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ConverterTaskParams(Param):
    """Encapsulates the parameters for the ConverterTask."""

    n: int = None
    frac: float = None
    n_jobs: int = 6
    random_state: int = None

    def __post_init__(self) -> None:
        if (self.n is not None and self.n != "NoneType") and (
            self.frac is not None and self.frac != "NoneType"
        ):
            msg = "Cannot provide both n and frac. One, the other or both must be None"
            logging.exception(msg)
            raise ValueError(msg)


# ------------------------------------------------------------------------------------------------ #
#                                CONVERTER TASK                                                    #
# ------------------------------------------------------------------------------------------------ #
class ConverterTask(Task):
    """Converts DICOM images to PNG Format

    Args:
        task_params (ConverterTaskParams): Parameters that control the task behavior.
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
        self.start()
        source_image_metadata = self._get_source_image_metadata()
        self._process_images(image_metadata=source_image_metadata)
        self.stop()

    def _get_source_image_metadata(self) -> pd.DataFrame:
        """Performs multivariate stratified sampling to obtain a fraction of the raw images."""

        # Read the raw DICOM metadata
        df = pd.read_csv(self._config.get_metadata_filepath())

        # Extract full mammogram images.
        image_metadata = df.loc[df["series_description"] == "full mammogram images"]

        # Define the stratum for stratified sampling
        stratum = ["image_view", "abnormality_type", "cancer", "assessment"]

        # Execute the sampling and obtain the case_ids
        df = image_metadata.groupby(by=stratum).sample(
            n=self.task_params.n,
            frac=self.task_params.frac,
            random_state=self.task_params.random_state,
        )

        return df

    def _process_images(self, image_metadata: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            image_metadata (pd.DataFrame): DataFrame containing image metadata.
        """
        for _, metadata in tqdm(image_metadata.iterrows(), total=image_metadata.shape[0]):
            self._process_image(metadata=metadata)

    @counter
    def _process_image(self, metadata: pd.Series) -> None:
        # Read the pixel data from DICOM files
        pixel_data = self._io.read(filepath=metadata["filepath"])
        # Execute the method to convert the data to 8-bit grayscale
        pixel_data = self._method().execute(image=pixel_data)
        # Create an image object
        image = ImageFactory.create(
            case_id=metadata["case_id"],
            stage_id=self._stage.uid,
            pixel_data=pixel_data,
            method=self._method.__name__,
            task_id=self._uid,
        )
        # Persist
        self.uow.image_repo.add(image=image)
