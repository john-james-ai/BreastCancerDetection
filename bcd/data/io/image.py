#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/io/image.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 11:47:17 am                                              #
# Modified   : Saturday October 21st 2023 01:22:59 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import sys
from dotenv import load_dotenv
import logging

import pydicom
import cv2
import numpy as np

load_dotenv()
# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class ImageIO:
    def read(self, filepath) -> np.ndarray:
        filepath = os.path.abspath(filepath)
        if "dcm" in filepath:
            return self._read_dicom(filepath=filepath)
        else:
            return self._read_image(filepath=filepath)

    def write(self, pixel_data: np.ndarray, filepath: str) -> None:
        filepath = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, pixel_data)

    def get_filepath(self, state: str, case_id: str, format: str) -> str:
        basedir = os.getenv(state, None)
        if not basedir:
            msg = f"{state} is invalid. Valid values are 'raw', 'dev', and 'final'."
            raise ValueError(msg)
        filename = case_id + "." + format
        return os.path.join(basedir, filename)

    def _read_dicom(self, filepath: str) -> np.ndarray:
        ds = pydicom.dcmread(filepath)
        return ds.pixel_array

    def _read_image(self, filepath: str) -> np.ndarray:
        return cv2.imread(filename=filepath)
