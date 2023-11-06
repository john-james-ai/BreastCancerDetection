#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/io/image_io.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 11:47:17 am                                              #
# Modified   : Sunday November 5th 2023 05:45:40 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import pandas as pd
import pydicom

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
    def get_filepath(cls, uid: str, fileset: str, label: bool, fileformat: str) -> str:
        """Returns a filepath based upon the fileset, label, and fileformat.

        Args:
            uid (str): The unique identifier for the image.
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
        return os.path.join(basedir, fileset, label, filename)

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
