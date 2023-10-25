#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/manage_data/io/image.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 11:47:17 am                                              #
# Modified   : Wednesday October 25th 2023 04:55:00 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import logging

import pydicom
import cv2
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class ImageIO:
    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def read(self, filepath) -> np.ndarray:
        filepath = self._parse_filepath(filepath=filepath)
        if "dcm" in filepath:
            return self._read_dicom(filepath=filepath)
        else:
            return self._read_image(filepath=filepath)

    def write(self, pixel_data: np.ndarray, filepath: str) -> None:
        filepath = self._parse_filepath(filepath=filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, pixel_data)

    def get_filepath(self, id: str, basedir: str, format: str) -> str:
        filename = id + "." + format
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
