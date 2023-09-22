#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/case/dataset.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:24:00 am                                              #
# Modified   : Friday September 22nd 2023 03:32:52 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Dataset Module"""
import sys
import os
import logging

import pandas as pd

from bcd.data.base import CBISDataset

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------------------------ #
CALC_DTYPES = {
    "patient_id": "str",
    "breast_density": "category",
    "left_or_right_breast": "category",
    "image_view": "category",
    "abnormality_id": "category",
    "abnormality_type": "category",
    "calc_type": "category",
    "calc_distribution": "category",
    "assessment": "category",
    "pathology": "category",
    "subtlety": "category",
    "dataset": "category",
    "image_series_uid": "str",
    "roi_mask_series_uid": "str",
    "cropped_image_series_uid": "str",
}
MASS_DTYPES = {
    "patient_id": "str",
    "breast_density": "category",
    "left_or_right_breast": "category",
    "image_view": "category",
    "abnormality_id": "category",
    "abnormality_type": "category",
    "mass_shape": "category",
    "mass_margins": "category",
    "assessment": "category",
    "pathology": "category",
    "subtlety": "category",
    "dataset": "category",
    "image_series_uid": "str",
    "roi_mask_series_uid": "str",
    "cropped_image_series_uid": "str",
}


# ------------------------------------------------------------------------------------------------ #
class MassCaseDataset(CBISDataset):
    """Dataset containing mass cases

    Args:
        filepath (str): File path to the dataset
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        df = pd.read_csv(self._filepath, dtype=MASS_DTYPES)
        super().__init__(df=df)


# ------------------------------------------------------------------------------------------------ #
class CalcCaseDataset(CBISDataset):
    """Dataset containing calcification cases

    Args:
        filepath (str): File path to the dataset
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        df = pd.read_csv(self._filepath, dtype=CALC_DTYPES)
        super().__init__(df=df)
