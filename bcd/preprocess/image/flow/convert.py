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
# Modified   : Wednesday November 1st 2023 01:49:36 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

import pandas as pd

from bcd.config import Config
from bcd.core.base import Param
from bcd.preprocess.image.flow.basetask import Task


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ImageConverterParams(Param):
    """Defines the parameters for the Converter Task"""

    frac: float = 0.005
    random_state: int = None


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Converter(Task):
    """Converts DICOM images to PNG Format"""

    params: ImageConverterParams

    def run(self) -> None:
        self.images_processed = 0

        source_image_metadata = self._get_source_image_metadata()

    def _get_source_image_metadata(self) -> pd.DataFrame:
        """Performs multivariate stratified sampling to obtain a fraction of the raw images."""

        # Read the raw DICOM metadata
        df = pd.read_csv(Config.get_metadata_filepath())

        # Extract full mammogram images.
        image_metadata = df.loc[df["series_description"] == "full mammogram images"]

        # Define the stratum for stratified sampling
        stratum = ["image_view", "abnormality_type", "cancer", "assessment"]

        # Execute the sampling and obtain the case_ids
        df = image_metadata.groupby(by=stratum).sample(
            frac=self.params.frac, random_state=self.params.random_state
        )

        return df

    def _process_images(self, image_metadata: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            image_metadata (pd.DataFrame): DataFrame containing image metadata.
        """
        for _, metadata in tqdm(image_metadata.iterrows(), total=image_metadata.shape[0]):
            pixel_data = self.read_pixel_data(filepath=metadata["filepath"])
            pixel_data = self._to_grayscale(pixel_data=pixel_data)
            image = self.create_image(case_id=metadata["case_id"], pixel_data=pixel_data)
            self.save_image(image=image)
            self._images_processed += 1
