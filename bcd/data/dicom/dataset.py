#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/dicom/dataset.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:24:00 am                                              #
# Modified   : Saturday September 23rd 2023 12:51:41 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Dataset Module"""
import sys
import os
import logging

import pandas as pd

from bcd.data.dataset import Dataset

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------------------------ #
DICOM_DTYPES = {
    "patient_id": "str",
    "series_uid": "str",
    "side": "category",
    "image_view": "category",
    "height": "int32",
    "width": "int32",
    "bits": "category",
    "smallest_image_pixel": "int32",
    "largest_image_pixel": "int32",
    "image_pixel_range": "int32",
}


# ------------------------------------------------------------------------------------------------ #
class DicomDataset(Dataset):
    """Dataset containing dicom image metadata

    Args:
        filepath (str): File path to the dataset
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        df = pd.read_csv(self._filepath, dtype=DICOM_DTYPES)
        super().__init__(df=df)
