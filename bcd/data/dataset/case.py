#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/dataset/case.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:24:00 am                                              #
# Modified   : Thursday September 28th 2023 07:17:24 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Dataset Module"""
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
CASE_DTYPES = {
    "case_id": "str",
    "patient_id": "str",
    "breast_density": "int32",
    "left_or_right_breast": "category",
    "image_view": "category",
    "abnormality_id": "int32",
    "abnormality_type": "category",
    "calc_type": "category",
    "calc_distribution": "category",
    "mass_shape": "category",
    "mass_margins": "category",
    "assessment": "int32",
    "pathology": "category",
    "subtlety": "int32",
    "dataset": "category",
}


# ------------------------------------------------------------------------------------------------ #
class CaseDataset(Dataset):
    """Dataset containing mass cases

    Args:
        filepath (str): File path to the dataset
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        df = pd.read_csv(self._filepath, dtype=CASE_DTYPES)
        super().__init__(df=df)

    def summary(self) -> pd.DataFrame:  # noqa
        """Summarizes the case dataset"""
        d = {}
        d["Patients"] = self._df["patient_id"].nunique()
        d["Cases"] = self._df["case_id"].nunique()
        d["Calcification Cases"] = self._df.loc[
            self._df["abnormality_type"] == "calcification"
        ].shape[0]
        d["Calcification Cases - Benign"] = self._df.loc[
            (self._df["abnormality_type"] == "calcification")
            & (self._df["cancer"] == False)  # noqa
        ].shape[0]
        d["Calcification Cases - Malignant"] = self._df.loc[
            (self._df["abnormality_type"] == "calcification") & (self._df["cancer"] == True)  # noqa
        ].shape[0]

        d["Mass Cases"] = self._df.loc[self._df["abnormality_type"] == "mass"].shape[0]
        d["Mass Cases - Benign"] = self._df.loc[
            (self._df["abnormality_type"] == "mass") & (self._df["cancer"] == False)  # noqa
        ].shape[0]
        d["Mass Cases - Malignant"] = self._df.loc[
            (self._df["abnormality_type"] == "mass") & (self._df["cancer"] == True)  # noqa
        ].shape[0]
        df = pd.DataFrame(data=d, index=[0]).T
        df.columns = ["Summary"]
        return df
