#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/etl/load.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 31st 2023 04:45:05 am                                               #
# Modified   : Monday November 6th 2023 01:19:39 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Load Task Module"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from dependency_injector.wiring import Provide, inject
from tqdm import tqdm

from bcd import Stage
from bcd.config import Config
from bcd.container import BCDContainer
from bcd.dal.io.image_io import ImageIO
from bcd.dal.repo.base import Repo
from bcd.image import ImageFactory


# ------------------------------------------------------------------------------------------------ #
#                                CONVERTER TASK                                                    #
# ------------------------------------------------------------------------------------------------ #
class Loader:
    """Converts DICOM images to PNG Format and loads them into the repository.

    Args:
        n (int): Number of image to load. Can't be used with frac. If groupby parameter
            is provided, this will be the number of images in each group.
        frac (float): The proportion of images to load. Can't be used with n. If
            the groupby parameter is provided, this will be the proportion
            of each group loaded.
        config (type[Config]): The application configuration class
        groupby (list): List of grouping variables. The default is image view, abnormality type,
            assessment and cancer diagnosis.
        io (type[ImageIO]): The class responsible for image io
        random_state (int): Seed for pseudo random sampling.
        force (bool): Whether to force a new load if images have already been loaded.
    """

    _stage = Stage(uid=0)

    @inject
    def __init__(
        self,
        n: int = None,
        frac: float = None,
        groupby: list = None,
        config: type[Config] = Config,
        io: type[ImageIO] = ImageIO,
        repo: Repo = Provide[BCDContainer.dal.image_repo],
        random_state: int = None,
        force: bool = False,
    ) -> None:
        self._n = n
        self._frac = frac
        self._groupby = groupby
        self._config = config
        self._io = io
        self._repo = repo
        self._random_state = random_state
        self._force = force

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

        if n is not None and frac is not None:
            msg = "Both n and frac cannot be provided. Provide n or frac, not both."
            self._logger.exception(msg)
            raise ValueError(msg)

    def run(self) -> None:
        """Converts the DICOM data to png and loads the repository with the new images."""
        if not self._images_exist() or self._force:
            # Delete all images for this stage (and mode)
            # This will only delete images for this stage and
            # the current mode, beit 'test', 'dev', 'exp', or 'prod.
            self._repo.delete_by_stage(stage_id=self._stage.uid)
            # Obtains the metadata, if sampled, stratified as described above.
            source_image_metadata = self._get_source_image_metadata()
            # Iterate through the images.
            self._process_images(image_metadata=source_image_metadata)
        else:
            msg = "Task aborted. Images exist."
            self._logger.info(msg)

    def _get_source_image_metadata(self) -> pd.DataFrame:
        """Performs multivariate stratified sampling to obtain a fraction of the raw images."""

        # Read the raw DICOM metadata
        df = pd.read_csv(self._config.get_dicom_metadata_filepath())

        # Extract full mammogram images.
        image_metadata = df.loc[df["series_description"] == "full mammogram images"]

        # Define the stratum for stratified sampling
        groupby = self._groupby or [
            "image_view",
            "abnormality_type",
            "cancer",
            "assessment",
        ]

        # Execute the sampling and obtain the case_ids
        df = image_metadata.groupby(by=groupby).sample(
            n=self._n,
            frac=self._frac,
            random_state=self._random_state,
        )

        return df

    def _process_images(self, image_metadata: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            image_metadata (pd.DataFrame): DataFrame containing image metadata.
        """
        for _, metadata in tqdm(image_metadata.iterrows(), total=image_metadata.shape[0]):
            self._process_image(metadata=metadata)

    def _process_image(self, metadata: pd.Series) -> None:
        start = datetime.now()
        # Read the pixel data from DICOM files
        image = self._io.read(filepath=metadata["filepath"])
        # Convert to float to avoid overflow or underflow.
        image = image.astype(float)
        # Rescale to gray scale values between 0-255
        img_gray = (image - image.min()) / (image.max() - image.min()) * 255.0
        # Convert to uint
        img_gray = np.uint8(img_gray)
        # Capture build time
        stop = datetime.now()
        build_time = (stop - start).total_seconds()
        # Create an image object
        image = ImageFactory.create(
            case_id=metadata["case_id"],
            stage_id=self._stage.uid,
            pixel_data=img_gray,
            method="Loader",
            build_time=build_time,
        )
        # Persist
        self._repo.add(image=image)

    def _images_exist(self) -> bool:
        """Returns True if images exist."""
        condition = lambda df: df["stage_id"] == self._stage.uid
        count = self._repo.count(condition)
        return count > 0
