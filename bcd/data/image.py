#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/image.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 11:47:17 am                                              #
# Modified   : Thursday December 14th 2023 09:42:18 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
import os
import shutil
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from bcd.config import Config


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
#                                      IMAGE IO                                                    #
# ------------------------------------------------------------------------------------------------ #
class ImageIO:
    """Manages IO of Image objects."""

    @classmethod
    def read(cls, filepath: str, channels: int = 2) -> np.ndarray:
        """Reads an image in DICOM, png, or jpeg format

        Args:
            filepath (str): Path to the file
            channels (int): Number of channels to read. This applies to non-DICOM
             images only.
        """
        filepath = cls._parse_filepath(filepath=filepath)
        if "dcm" in filepath:
            return cls._read_dicom(filepath=filepath)
        else:
            return cls._read_image(filepath=filepath, channels=channels)

    @classmethod
    def write(cls, pixel_data: np.ndarray, filepath: str, force: bool = False) -> None:
        filepath = cls._parse_filepath(filepath=filepath)
        if force or not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # pylint: disable=no-member
            cv2.imwrite(filepath, pixel_data)
        elif os.path.exists(filepath):
            msg = f"Image filename {os.path.realpath(filepath)} \
            already exists. If you wish to overwrite, set force = True"
            logging.warning(msg)

    @classmethod
    def delete(cls, filepath: str, silent: bool = False) -> None:
        """Deletes an image"""
        try:
            os.remove(filepath)
        except OSError:
            if not silent:
                msg = f"Delete warning! No image exists at {filepath}."
                logging.warning(msg)

    @classmethod
    def build_filepath(
        cls, uid: str, stage: str, fileset: str, label: bool, fileformat: str
    ) -> str:
        """Constructs a filepath for the current mode, stage, fileset, and label

        Args:
            uid (str): The unique identifier for the image.
            stage (str): The stage of the image processing cycle.
            fileset (str): Either 'train' or 'test'
            label (bool): True for malignancy, False for benign
            fileformat (str): Format of image on disk.
        """
        basedir = Config.get_data_dir()
        if label is True or label == 1 or label == "TRUE":
            label = "malignant"
        else:
            label = "benign"
        filename = uid + "." + fileformat
        return os.path.join(basedir, stage, fileset, label, filename)

    @classmethod
    def _read_dicom(cls, filepath: str) -> np.ndarray:
        try:
            ds = pydicom.dcmread(filepath)
        except Exception as e:
            logging.exception(e)
            raise
        else:
            return ds.pixel_array

    @classmethod
    def _read_image(cls, filepath: str, channels: int = 2) -> np.ndarray:
        try:
            if channels == 2:
                return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            else:
                return cv2.imread(filepath)
        except FileNotFoundError as e:
            logging.exception(e)
            raise
        except Exception as e:
            logging.exception(e)
            raise

    @classmethod
    def _parse_filepath(cls, filepath: str) -> str:
        if isinstance(filepath, pd.Series):
            filepath = filepath.values[0]
        filepath = os.path.abspath(filepath)
        return filepath


# ------------------------------------------------------------------------------------------------ #
#                                      IMAGE CONVERTER                                             #
# ------------------------------------------------------------------------------------------------ #
class ImageConverter:
    """Converts all or a sample of DICOM images to PNG format.

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

    __stage = "converted"

    def __init__(
        self,
        n: int = None,
        frac: float = None,
        groupby: list = None,
        config: type[Config] = Config,
        io: type[ImageIO] = ImageIO,
        random_state: int = None,
        force: bool = False,
    ) -> None:
        self._n = n
        self._frac = frac
        self._groupby = groupby
        self._config = config
        self._io = io
        self._random_state = random_state
        self._force = force

        self._images_converted = 0

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

        if n is not None and frac is not None:
            msg = "Both n and frac cannot be provided. Provide n or frac, not both."
            self._logger.exception(msg)
            raise ValueError(msg)

    @property
    def destination(self) -> str:
        return self._get_destination()

    @property
    def images_converted(self) -> int:
        return self._images_converted

    def run(self) -> None:
        """Converts the DICOM data to png and loads the repository with the new images."""

        # Raise FileExists exception if destination already exists and Force is False.
        destination = self._get_destination()

        if os.path.exists(destination) and not self._force:
            msg = f"Task aborted. Images exist at {destination}."
            self._logger.info(msg)
            raise FileExistsError(msg)

        # Delete any existing images for the current mode and stage.
        shutil.rmtree(destination, ignore_errors=True)

        # Obtains the metadata, if sampled, stratified as described above.
        source_image_metadata = self._get_source_image_metadata()

        # Iterate through the images.
        self._process_images(image_metadata=source_image_metadata)

    def _get_source_image_metadata(self) -> pd.DataFrame:
        """Performs multivariate stratified sampling to obtain a fraction of the raw images."""

        # Read the raw DICOM metadata
        df = pd.read_csv(self._config.get_dicom_metadata_filepath())

        # Extract full mammogram images.
        image_metadata = df.loc[df["series_description"] == "full mammogram images"]

        # Define the stratum for stratified sampling
        groupby = self._groupby or [
            "fileset",
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

    def _get_destination(self) -> str:
        basedir = Config.get_data_dir()
        return basedir + "/" + self.__stage + "/"

    def _process_images(self, image_metadata: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            image_metadata (pd.DataFrame): DataFrame containing image metadata.
        """
        for _, metadata in tqdm(
            image_metadata.iterrows(), total=image_metadata.shape[0]
        ):
            self._process_image(metadata=metadata)

    def _process_image(self, metadata: pd.Series) -> None:
        # Read the pixel data from DICOM files
        image = self._io.read(filepath=metadata["filepath"])
        # Convert to float to avoid overflow or underflow.
        image = image.astype(float)
        # Rescale to gray scale values between 0-255
        img_gray = (image - image.min()) / (image.max() - image.min()) * 255.0
        # Convert to uint
        img_gray = np.uint8(img_gray)
        # Obtain a filepath for the mode, stage, fileset, and label.
        filepath = self._io.build_filepath(
            uid=str(uuid4()),
            stage=self.__stage,
            fileset=metadata["fileset"],
            label=metadata["cancer"],
            fileformat="png",
        )

        self._io.write(pixel_data=img_gray, filepath=filepath)
        self._images_converted += 1
