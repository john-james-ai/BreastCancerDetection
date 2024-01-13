#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/image.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 11:47:17 am                                              #
# Modified   : Thursday January 11th 2024 03:58:31 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
import os
from typing import Callable

import cv2
import dicomsdl as dicom
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------ #
METADATA_FILEPATH = "data/meta/3_clean/cbis.csv"


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
    def _read_dicom(cls, filepath: str) -> np.ndarray:
        try:
            ds = dicom.open(filepath)
        except Exception as e:
            logging.exception(e)
            raise
        else:
            return ds.pixelData()

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
#                                      IMAGE REPO                                                  #
# ------------------------------------------------------------------------------------------------ #
class ImageRepo:
    """Provides a repository interface to Images

    Args:
        io (type[ImageIO]): ImageIO class type
        force (bool): Overwrites existing data only if True. Default = False
    """

    def __init__(self, io: type[ImageIO] = ImageIO, force: bool = False) -> None:
        self._io = io
        self._force = force
        self._meta = pd.read_csv(METADATA_FILEPATH)
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def force(self) -> bool:
        return self._force

    def query(
        self,
        n: int = None,
        frac: float = None,
        condition: Callable = None,
        groupby: list = None,
        columns: list = None,
        random_state: int = None,
    ) -> pd.DataFrame:
        """Returns the raw image metadata matching the query parameters.

        Args:
            n (int): Number of records to return or the number of records
                to return in each group, if groupby is not None. Default = None
            frac (float): The fraction of all records to return or the
                fraction of records by group if groupby is not None. Ignored
                if n is not None. Default = None
            condition (Callable): Lambda expression used to subset the metadata.
                Default is None
            groupby (list): List of grouping variables for stratified sampling.
                Default is None.
            columns (list): List of columns to select.
            random_state (int): Seed for pseudo-randomization

            If all parameters are None, all records are returned.
        """
        if n is not None and frac is not None:
            msg = "Ambiguous parameters. n or frac must be None. "
            self._logger.exception(msg)
            raise ValueError(msg)

        df = self._meta.copy(deep=True)

        if condition is not None:
            try:
                df = df[condition].copy(deep=True)
            except KeyError as exc:
                msg = "Invalid condition. "
                self._logger.exception(msg)
                raise ValueError(msg) from exc

        if groupby is not None:
            df = df.groupby(by=groupby).sample(
                n=n, frac=frac, random_state=random_state
            )
        elif n is not None or frac is not None:
            df = df.sample(n=n, frac=frac, random_state=random_state)

        if columns is not None:
            df = df[columns].copy(deep=True)

        df = df.drop_duplicates()

        return df

    def get(self, filepath: str) -> np.ndarray:
        """Returns an image from the repository

        Args:
            filepath (str): Path to the image.

        Raises: FileNotFound Exception if file not found.
        """
        return self._io.read(filepath=filepath)

    def add(
        self,
        image: np.ndarray,
        destination: str,
        fileset: str,
        label: bool,
        filename: str,
    ) -> None:
        """Adds an image to the repository.

        Args:
            image (np.ndarray): Image in numpy array format.
            destination (str): The parent directory for the data.
            fileset (str): Whether the image is from the 'training' or 'test' set
            label (bool): Whether the image represents a malignancy
            filename (str): Name of the file

        Raises FileExistError if file already exists and Force is False
        """
        filepath = self._get_filepath(
            destination=destination, fileset=fileset, label=label, filename=filename
        )
        if os.path.exists(filepath) and not self._force:
            msg = f"{filepath} already exists."
            self._logger.exception(msg)
            raise FileExistsError(msg)
        else:
            self._io.write(pixel_data=image, filepath=filepath, force=self._force)

    def exists(
        self, destination: str, fileset: str, label: bool, filename: str
    ) -> bool:
        """Evaluates existence of an image as specified.

        Args:
            destination (str): The parent directory for the data.
            fileset (str): Whether the image is from the 'training' or 'test' set
            label (bool): Whether the image represents a malignancy
            filename (str): Name of the file


        """
        return os.path.exists(
            self._get_filepath(
                destination=destination, fileset=fileset, label=label, filename=filename
            )
        )

    def _get_filepath(
        self, destination: str, fileset: str, label: bool, filename: str
    ) -> str:
        pathology = "benign" if not label else "malignant"
        return os.path.join(destination, fileset, pathology, filename)
