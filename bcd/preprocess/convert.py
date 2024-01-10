#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/convert.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday January 9th 2024 12:44:48 am                                                #
# Modified   : Tuesday January 9th 2024 07:36:59 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
# pylint: disable=no-member, line-too-long, import-error
# ------------------------------------------------------------------------------------------------ #
"""Image Conversion Module"""
import logging
import os
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from bcd.dal.image import ImageIO


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, line-too-long, import-error
# ------------------------------------------------------------------------------------------------ #
#                                   DICOM IMAGE CONVERTER                                          #
# ------------------------------------------------------------------------------------------------ #
class DICOMImageConverter:
    """Converts all or a sample of DICOM images to the designated format. Currently supports png and jpeg.

    This class converts DICOM images to png or jpeg format. Minimal change

    Args:
        metafilepath (str): Filepath containing image metadata.
        destination (str): Directory into which the converted files will be stored.
        registry_filepath (str): Filepath to the registry.
        fmt (str): Either 'png', or 'jpeg'
        bits (int): Either 8 or 16 bits unsigned. Pixel values are rescaled accordingly.
        n (int): Number of image to load. Can't be used with frac. If groupby parameter
            is provided, this will be the number of images in each group.
        frac (float): The proportion of images to load. Can't be used with n. If
            the groupby parameter is provided, this will be the proportion
            of each group loaded.
        condition (Callable): Lambda callable statement that can be used to filter the data.
        groupby (list): List of grouping variables. The default = None
        multiclass (bool): Whether to structure the data for multiclass or binary classification. In
            multiclass classification, the data will be separated into the following subdirectories:
                benign_calc
                benign_mass
                malignant_calc
                malignant_mass
            For binary classification, the subdirectories will be 'benign' or 'malignant'. The
            default is False.
        io (type[ImageIO]): The class responsible for image io
        random_state (int): Seed for pseudo random sampling.
        force (bool): Whether to force a new load if images have already been loaded.
    """

    def __init__(
        self,
        metafilepath: str,
        destination: str,
        registry_filepath: str,
        fmt: str = "png",
        bits: int = 16,
        n: int = None,
        frac: float = None,
        condition: Callable = None,
        groupby: list = None,
        multiclass: bool = False,
        io: type[ImageIO] = ImageIO,
        n_jobs: int = 12,
        random_state: int = None,
        force: bool = False,
    ) -> None:
        self._metafilepath = os.path.abspath(metafilepath)
        self._destination = os.path.abspath(destination)
        self._registry_filepath = os.path.abspath(registry_filepath)
        self._fmt = fmt
        self._bits = bits
        self._n = n
        self._frac = frac
        self._condition = condition
        self._groupby = groupby
        self._multiclass = multiclass
        self._io = io
        self._n_jobs = n_jobs
        self._random_state = random_state
        self._force = force

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def destination(self) -> str:
        return self._destination

    def convert(self) -> None:
        """Converts the DICOM data to png and loads the repository with the new images."""

        self._validate()
        # Obtains the metadata, if sampled, stratified as described above.
        source_image_metadata = self._get_source_image_metadata()

        # Iterate through the images.
        destination_image_metadata = self._process_images(
            image_metadata=source_image_metadata
        )

        # Save registry
        self._save_registry(destination_image_metadata)

    def _validate(self) -> bool:
        if self._n is not None and self._frac is not None:
            msg = "Both n and frac cannot be provided. Provide n or frac, not both."
            self._logger.exception(msg)
            raise ValueError(msg)
        if not os.path.exists(self._metafilepath):
            msg = f"No metadata file exists at {self._metafilepath}"
            self._logger.exception(msg)
            raise ValueError(msg)
        if self._fmt not in ["png", "jpeg"]:
            msg = f"Format {self._fmt} is not supported"
            self._logger.exception(msg)
            raise ValueError(msg)
        if self._bits not in [8, 16]:
            msg = f"Bit depths of 8 or 16 are supported, {self._bits} bits are not supported."
            self._logger.exception(msg)
            raise ValueError(msg)

    def _get_source_image_metadata(self) -> pd.DataFrame:
        """Performs multivariate stratified sampling to obtain a fraction of the raw images."""

        # Read the raw image metadata
        df = pd.read_csv(self._metafilepath)

        # Since we are only concerned with full mammogram imaging, abnormality level
        # information can be suppressed and deduplicated
        df = df.drop(
            columns=[
                "abnormality_id",
                "calc_type",
                "calc_distribution",
                "mass_shape",
                "mass_margins",
                "assessment",
            ]
        )
        df = df.drop_duplicates()

        # Filter before groupby and sampling
        if self._condition is not None:
            df = df[self._condition]

        if self._groupby is not None:
            df = df.groupby(by=self._groupby).sample(
                n=self._n, frac=self._frac, random_state=self._random_state
            )
        elif self._n is not None:
            df = df.sample(n=self._n, random_state=self._random_state)

        return df

    def _process_images(self, image_metadata: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            image_metadata (pd.DataFrame): DataFrame containing image metadata.
        """

        image_meta = joblib.Parallel(n_jobs=self._n_jobs)(
            joblib.delayed(self._process_image)(metadata)
            for _, metadata in tqdm(
                image_metadata.iterrows(), total=image_metadata.shape[0]
            )
        )
        image_meta_df = pd.DataFrame(data=image_meta)
        return image_meta_df

    def _process_image(self, metadata: pd.Series) -> None:
        """Process an image from a row of the metadata"""
        # Convert cancer to a pathology and create class label
        pathology = "malignant" if metadata["cancer"] else "benign"
        if self._multiclass:
            class_label = pathology + "_" + metadata["abnormality_type"][0:4]
        else:
            class_label = pathology

        # Create filename from mammogram id variables
        filename = (
            metadata["fileset"].capitalize()
            + "_"
            + metadata["abnormality_type"][0:4].capitalize()
            + "_"
            + metadata["patient_id"]
            + "_"
            + metadata["laterality"]
            + "_"
            + metadata["image_view"]
            + "_"
            + metadata["pathology"]
            + "."
            + self._fmt
        )

        # The filepath is consistent with TensorFlow image dataset generation
        # modules that expect images to be organized by class label.
        filepath = os.path.join(self._destination, class_label, filename)

        self._convert_image(
            source_filepath=metadata["filepath"], destination_filepath=filepath
        )

        # Update metadata with new filepath and class label information
        meta = metadata.copy(deep=True)
        meta["class_label"] = class_label
        meta["filepath"] = filepath
        meta["file_size"] = os.path.getsize(filepath)
        meta["bit_depth"] = self._bits
        return meta

    def _convert_image(self, source_filepath: str, destination_filepath: str) -> None:
        """Converts the image if needed.

        If the image doesn't already exist, it is converted to the target format. If
        the image already exists, it is converted only if force is True.
        """
        if (os.path.exists(destination_filepath) and self._force) or not os.path.exists(
            destination_filepath
        ):
            # Read the pixel data from DICOM files
            image = self._io.read(filepath=source_filepath)

            # Rescale and convert to the target bit_depth as in the bits parameter.
            image = image.astype(float)
            image = (
                (image - image.min())
                / (image.max() - image.min())
                * (2**self._bits - 1)
            )
            if self._bits == 8:
                image = np.uint8(image)
            elif self._bits == 16:
                image = np.uint16(image)

            self._io.write(
                pixel_data=image, filepath=destination_filepath, force=self._force
            )

    def _save_registry(self, registry: pd.DataFrame) -> None:
        """Save registry.

        If the registry exists and force is True overwrite the existing registry.
        If the registry exists and not force, append to end of registry.
        """
        os.makedirs(os.path.dirname(self._registry_filepath), exist_ok=True)
        if os.path.exists(self._registry_filepath):
            existing_registry = pd.read_csv(self._registry_filepath)
            registry = pd.concat([existing_registry, registry])
            registry = registry.drop_duplicates()

        registry.to_csv(self._registry_filepath, mode="w", index=False)
