#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/convert.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 09:59:41 pm                                                #
# Modified   : Sunday October 29th 2023 02:04:31 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Converts DICOM Data to PNG Format"""
from dataclasses import dataclass

import pandas as pd
from dependency_injector.wiring import Provide, inject
from dotenv import load_dotenv
from tqdm import tqdm

from bcd.container import BCDContainer
from bcd.core.base import Param
from bcd.preprocess.image.base import Transformer

# ------------------------------------------------------------------------------------------------ #
load_dotenv()


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ImageConverterParams(Param):
    """Defines the parameters for the Converter Task"""

    frac: float = 0.005
    random_state: int = None


# ------------------------------------------------------------------------------------------------ #
class ImageConverter(Transformer):
    """Converts the DICOM images to PNG format."""

    name = __qualname__
    module = __name__
    stage_id = 0

    @inject
    def __init__(
        self,
        task_id: str,
        params: Param,
        metadata_filepath: str = Provide[BCDContainer.config.data.metadata],
    ) -> None:
        super().__init__(task_id=task_id)
        self._params = params
        self._metadata_filepath = metadata_filepath
        self._images_processed = 0

    @property
    def images_processed(self) -> int:
        return self._images_processed

    def execute(self) -> None:
        """Converts DICOM images to PNG format."""
        source_image_metadata = self._get_source_image_metadata()
        self._process_images(image_metadata=source_image_metadata)

    def _get_source_image_metadata(self) -> pd.DataFrame:
        """Performs multivariate stratified sampling to obtain a fraction of the raw images."""

        # Read the raw DICOM metadata
        df = pd.read_csv(self._metadata_filepath)

        # Extract full mammogram images.
        image_metadata = df.loc[df["series_description"] == "full mammogram images"]

        # Define the stratum for stratified sampling
        stratum = ["image_view", "abnormality_type", "cancer", "assessment"]

        # Execute the sampling and obtain the case_ids
        df = image_metadata.groupby(by=stratum).sample(
            frac=self._params.frac, random_state=self._params.random_state
        )

        return df

    def _process_images(self, image_metadata: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            image_metadata (pd.DataFrame): DataFrame containing image metadata.
        """
        for _, metadata in tqdm(image_metadata.iterrows(), total=image_metadata.shape[0]):
            pixel_data = self.read_pixel_data(filepath=metadata["filepath"])
            image = self.create_image(case_id=metadata["case_id"], pixel_data=pixel_data)
            self.save_image(image=image)
            self._images_processed += 1