#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/dicom/dqa.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Friday September 22nd 2023 07:44:58 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Data Quality Module"""
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
class DicomDQA(DQA):
    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        self._df = pd.read_csv(self._filepath)

    @property
    def series_uid_validity(self) -> pd.Series:
        sop = "1.3.6.1.4.1.9590"
        n = self._n_rows(df=self._df)
        nv = self._n_values_containing(s=self._df["series_uid"], substr=sop)
        pv = self._p_values_containing(s=self._df["series_uid"], substr=sop)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def filepath_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_filepaths_exist(self._df["filepath"])
        pv = self._p_filepaths_exist(self._df["filepath"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def patient_id_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_containing(s=self._df["patient_id"], substr="P_")
        pv = self._p_values_containing(s=self._df["patient_id"], substr="P_")
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def side_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["side"], values=["LEFT", "RIGHT"])
        pv = self._p_values_valid(s=self._df["side"], values=["LEFT", "RIGHT"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def image_view_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["image_view"], values=["CC", "MLO"])
        pv = self._p_values_valid(s=self._df["image_view"], values=["CC", "MLO"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def height_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(
            s=self._df["height"].astype("int32"), min_value=1, max_value=10000
        )
        pv = self._p_within_range(
            s=self._df["height"].astype("int32"), min_value=1, max_value=10000
        )
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def width_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(s=self._df["width"].astype("int32"), min_value=1, max_value=10000)
        pv = self._p_within_range(s=self._df["width"].astype("int32"), min_value=1, max_value=10000)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def bits_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["bits"].astype("int32"), values=[8, 16])
        pv = self._p_values_valid(s=self._df["bits"].astype("int32"), values=[8, 16])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def smallest_image_pixel_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(
            s=self._df["smallest_image_pixel"].astype("int64"), min_value=0, max_value=65535
        )
        pv = self._p_within_range(
            s=self._df["smallest_image_pixel"].astype("int64"), min_value=0, max_value=65535
        )
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def largest_image_pixel_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(
            s=self._df["largest_image_pixel"].astype("int64"), min_value=0, max_value=65535
        )
        pv = self._p_within_range(
            s=self._df["largest_image_pixel"].astype("int64"), min_value=0, max_value=65535
        )
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def image_pixel_range_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(
            s=self._df["image_pixel_range"].astype("int64"), min_value=0, max_value=65535
        )
        pv = self._p_within_range(
            s=self._df["image_pixel_range"].astype("int64"), min_value=0, max_value=65535
        )
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
        fp = self.filepath_validity
        pid = self.patient_id_validity
        side = self.side_validity
        view = self.image_view_validity
        height = self.height_validity
        width = self.width_validity
        bits = self.bits_validity
        sip = self.smallest_image_pixel_validity
        lip = self.largest_image_pixel_validity
        ipr = self.image_pixel_range_validity
        dfv = pd.concat(
            [suid, fp, pid, side, view, height, width, bits, sip, lip, ipr],
            axis=1,
        )
        dfv.columns = [
            "series_uid",
            "filepath",
            "patient_id",
            "side",
            "image_view",
            "height",
            "width",
            "bits",
            "smallest_image_pixel",
            "largest_image_pixel",
            "image_pixel_range",
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
