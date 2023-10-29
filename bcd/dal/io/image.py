#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/io/image.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 11:47:17 am                                              #
# Modified   : Sunday October 29th 2023 01:30:41 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
import os

import cv2
import numpy as np
import pandas as pd
import pydicom


# ------------------------------------------------------------------------------------------------ #
class ImageIO:
    """Manages IO of Image objects."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def read(self, filepath) -> np.ndarray:
        filepath = self._parse_filepath(filepath=filepath)
        if "dcm" in filepath:
            return self._read_dicom(filepath=filepath)
        else:
            return self._read_image(filepath=filepath)

    def write(self, pixel_data: np.ndarray, filepath: str, force: bool = False) -> None:
        filepath = self._parse_filepath(filepath=filepath)
        if force or not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # pylint: disable=no-member
            cv2.imwrite(filepath, pixel_data)
        elif os.path.exists(filepath):
            msg = f"Image filename {os.path.realpath(filepath)} \
            already exists. If you wish to overwrite, set force = True"
            self._logger.warning(msg)

    def delete(self, filepath: str, silent: bool = False) -> None:
        """Deletes an image"""
        try:
            os.remove(filepath)
        except OSError:
            if not silent:
                msg = f"Delete warning! No image exists at {filepath}."
                self._logger.warning(msg)

    def get_filepath(self, uid: str, basedir: str, fileformat: str) -> str:
        filename = uid + "." + fileformat
        return os.path.join(basedir, filename)

    def _read_dicom(self, filepath: str) -> np.ndarray:
        try:
            ds = pydicom.dcmread(filepath)
        except Exception as e:
            self._logger.exception(e)
            raise
        else:
            return ds.pixel_array

    def _read_image(self, filepath: str) -> np.ndarray:
        try:
            # pylint: disable=no-member
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        except FileNotFoundError as e:
            self._logger.exception(e)
            raise
        except Exception as e:
            self._logger.exception(e)
            raise
        else:
            return image

    def _parse_filepath(self, filepath: str) -> str:
        if isinstance(filepath, pd.Series):
            filepath = filepath.values[0]
        filepath = os.path.abspath(filepath)
        return filepath
