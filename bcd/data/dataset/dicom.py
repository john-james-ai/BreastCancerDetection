#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/dataset/dicom.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:24:00 am                                              #
# Modified   : Thursday September 28th 2023 04:25:59 am                                            #
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
    "series_uid": "str",
    "filepath": "str",
    "patient_id": "str",
    "side": "category",
    "image_view": "category",
    "photometric_interpretation": "category",
    "samples_per_pixel": "int32",
    "height": "int64",
    "width": "int64",
    "size": "int64",
    "aspect_ratio": "float",
    "bits": "category",
    "smallest_image_pixel": "int64",
    "largest_image_pixel": "int64",
    "image_pixel_range": "int64",
    "case_id": "str",
    "series_description": "str",
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

    def summary(self) -> pd.DataFrame:
        """Provides a summary of the DICOM Dataset"""
        df = self._df[
            [
                "series_description",
                "height",
                "width",
                "bits",
                "smallest_image_pixel",
                "largest_image_pixel",
                "image_pixel_range",
                "brisque",
            ]
        ]
        return df.groupby(by=["series_description"]).describe()
