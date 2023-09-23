#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/series/dqa.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:51 am                                              #
# Modified   : Friday September 22nd 2023 07:43:10 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Series Data Quality Module"""
import os
import sys
import logging

import pandas as pd

from bcd.data.base import DQA, DQAResult

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class SeriesDQA(DQA):
    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        self._df = pd.read_csv(self._filepath)

    @property
    def series_description_validity(self) -> pd.Series:
        series_desc = ["ROI mask images", "full mammogram images", "cropped images"]
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["series_description"], values=series_desc)
        pv = self._p_values_valid(s=self._df["series_description"], values=series_desc)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def series_uid_validity(self) -> pd.Series:
        sop = "1.3.6.1.4.1.9590"
        n = self._n_rows(df=self._df)
        nv = self._n_values_containing(s=self._df["series_uid"], substr=sop)
        pv = self._p_values_containing(s=self._df["series_uid"], substr=sop)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def number_of_images_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(
            s=self._df["number_of_images"].astype("int32"), min_value=1, max_value=2
        )
        pv = self._p_within_range(
            s=self._df["number_of_images"].astype("int32"), min_value=1, max_value=2
        )
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def file_location_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_filepaths_exist(self._df["file_location"])
        pv = self._p_filepaths_exist(self._df["file_location"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def validate(self) -> pd.DataFrame:
        "Validates the data and returns a boolean mask of cell validity."

    def analyze_validity(self) -> DQAResult:
        """Executes a Validity Assessment"""
        dvd = self._analyze_validity()
        dvs = self._summarize_validity(dfv=dvd)
        result = DQAResult(summary=dvs, detail=dvd)
        return result

    def _analyze_validity(self) -> pd.DataFrame:
        """Performs detailed validity assessment"""
        suid = self.series_uid_validity
        desc = self.series_description_validity
        nimages = self.number_of_images_validity
        floc = self.file_location_validity
        dfv = pd.concat(
            [
                suid,
                desc,
                nimages,
                floc,
            ],
            axis=1,
        )
        dfv.columns = [
            "series_uid",
            "series_description",
            "number_of_images",
            "file_location",
        ]
        return dfv.T

    def _summarize_validity(self, dfv: pd.DataFrame) -> pd.DataFrame:
        """Summarizes the validity assessment"""
        n = dfv["N"].sum()
        nv = dfv["Valid"].sum()
        pv = nv / n
        dv = {"N": n, "Valid": nv, "Validity": pv}
        dfv = pd.DataFrame(data=dv, index=[0]).T
        dfv.columns = ["Values"]
        return dfv
