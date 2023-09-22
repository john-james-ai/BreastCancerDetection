#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/case/dqa.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:51 am                                              #
# Modified   : Friday September 22nd 2023 10:19:06 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Data Quality Module"""
import sys
import os
from abc import abstractmethod
import logging

import pandas as pd

from bcd.data.base import DQA, DQAResult

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class CaseDQA(DQA):
    @abstractmethod
    def run(self) -> None:
        "Executes the data quality analysis"

    def _validate_patient_id(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_containing(s=self._df["patient_id"], substr="P_")
        pv = self._p_values_containing(s=self._df["patient_id"], substr="P_")
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_breast_density(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_range_valid(
            s=self._df["breast_density"].astype("int32"), min_value=1, max_value=4
        )
        pv = self._p_range_valid(
            s=self._df["breast_density"].astype("int32"), min_value=1, max_value=4
        )
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_left_or_right_breast(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["left_or_right_breast"], values=["LEFT", "RIGHT"])
        pv = self._p_values_valid(s=self._df["left_or_right_breast"], values=["LEFT", "RIGHT"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_image_view(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["image_view"], values=["CC", "MLO"])
        pv = self._p_values_valid(s=self._df["image_view"], values=["CC", "MLO"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_abnormality_id(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_range_valid(
            s=self._df["abnormality_id"].astype("int32"), min_value=1, max_value=10
        )
        pv = self._p_range_valid(
            s=self._df["abnormality_id"].astype("int32"), min_value=1, max_value=10
        )
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_abnormality_type(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["abnormality_type"], values=["calcification", "mass"])
        pv = self._p_values_valid(s=self._df["abnormality_type"], values=["calcification", "mass"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_assessment(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_range_valid(s=self._df["assessment"].astype("int32"), min_value=0, max_value=6)
        pv = self._p_range_valid(s=self._df["assessment"].astype("int32"), min_value=0, max_value=6)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_pathology(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["pathology"], values=["BENIGN", "MALIGNANT"])
        pv = self._p_values_valid(s=self._df["pathology"], values=["BENIGN", "MALIGNANT"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_subtlety(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_range_valid(s=self._df["subtlety"].astype("int32"), min_value=1, max_value=5)
        pv = self._p_range_valid(s=self._df["subtlety"].astype("int32"), min_value=1, max_value=5)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_dataset(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["dataset"], values=["train", "test"])
        pv = self._p_values_valid(s=self._df["dataset"], values=["train", "test"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_series_uid(self, uid: str) -> pd.Series:
        sop = "1.3.6.1.4.1.9590"
        n = self._n_rows(df=self._df)
        nv = self._n_values_containing(s=self._df[uid], substr=sop)
        pv = self._p_values_containing(s=self._df[uid], substr=sop)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)


# ------------------------------------------------------------------------------------------------ #
class CalcCaseDQA(CaseDQA):
    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        self._df = pd.read_csv(self._filepath)
        self._completeness = None
        self._uniqueness = None
        self._validity = None

    def run(self) -> None:
        "Executes the data quality analysis"

    def run_validity(self) -> DQAResult:
        """Executes a Validty Assessment"""
        dvd = self._run_validity_detail()
        dvs = self._run_validity_summary(dfv=dvd)
        result = DQAResult(summary=dvs, detail=dvd)
        return result

    def _run_validity_detail(self) -> pd.DataFrame:
        """Performs detailed validity assessment"""
        pid = self._validate_patient_id()
        bd = self._validate_breast_density()
        side = self._validate_left_or_right_breast()
        view = self._validate_image_view()
        aid = self._validate_abnormality_id()
        at = self._validate_abnormality_type()
        ct = self._validate_calc_type()
        cd = self._validate_calc_distribution()
        asmt = self._validate_assessment()
        path = self._validate_pathology()
        sbty = self._validate_subtlety()
        ds = self._validate_dataset()
        img_suid = self._validate_series_uid("image_series_uid")
        roi_suid = self._validate_series_uid("roi_mask_series_uid")
        crop_suid = self._validate_series_uid("cropped_image_series_uid")
        dfv = pd.concat(
            [
                pid,
                bd,
                side,
                view,
                aid,
                at,
                ct,
                cd,
                asmt,
                path,
                sbty,
                ds,
                img_suid,
                roi_suid,
                crop_suid,
            ],
            axis=1,
        )
        dfv.columns = [
            "patient_id",
            "breast_density",
            "side",
            "image_view",
            "abnormality_id",
            "abnormality_type",
            "calcification_type",
            "calcification_distribution",
            "assessment",
            "pathology",
            "subtlety",
            "dataset",
            "image_series_uid",
            "roi_mask_series_uid",
            "cropped_image_series_uid",
        ]
        return dfv.T

    def _run_validity_summary(self, dfv: pd.DataFrame) -> pd.DataFrame:
        """Summarizes the validity assessment"""
        n = dfv["N"].sum()
        nv = dfv["Valid"].sum()
        pv = nv / n
        dv = {"N": n, "Valid": nv, "Validity": pv}
        dfv = pd.DataFrame(data=dv, index=[0]).T
        dfv.columns = ["Values"]
        return dfv

    def _validate_calc_type(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_nonnull(s=self._df["calc_type"])
        pv = self._p_values_nonnull(s=self._df["calc_type"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_calc_distribution(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_nonnull(s=self._df["calc_distribution"])
        pv = self._p_values_nonnull(s=self._df["calc_distribution"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)


# ------------------------------------------------------------------------------------------------ #
class MassCaseDQA(CaseDQA):
    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        self._df = pd.read_csv(self._filepath)
        self._completeness = None
        self._uniqueness = None
        self._validity = None

    def run(self) -> None:
        "Executes the data quality analysis"

    def run_validity(self) -> DQAResult:
        """Executes a Validty Assessment"""
        dvd = self._run_validity_detail()
        dvs = self._run_validity_summary(dfv=dvd)
        result = DQAResult(summary=dvs, detail=dvd)
        return result

    def _run_validity_detail(self) -> pd.DataFrame:
        """Performs detailed validity assessment"""
        pid = self._validate_patient_id()
        bd = self._validate_breast_density()
        side = self._validate_left_or_right_breast()
        view = self._validate_image_view()
        aid = self._validate_abnormality_id()
        at = self._validate_abnormality_type()
        ms = self._validate_mass_shape()
        mm = self._validate_mass_margins()
        asmt = self._validate_assessment()
        path = self._validate_pathology()
        sbty = self._validate_subtlety()
        ds = self._validate_dataset()
        img_suid = self._validate_series_uid("image_series_uid")
        roi_suid = self._validate_series_uid("roi_mask_series_uid")
        crop_suid = self._validate_series_uid("cropped_image_series_uid")
        dfv = pd.concat(
            [
                pid,
                bd,
                side,
                view,
                aid,
                at,
                ms,
                mm,
                asmt,
                path,
                sbty,
                ds,
                img_suid,
                roi_suid,
                crop_suid,
            ],
            axis=1,
        )
        dfv.columns = [
            "patient_id",
            "breast_density",
            "side",
            "image_view",
            "abnormality_id",
            "abnormality_type",
            "mass_shape",
            "mass_margins",
            "assessment",
            "pathology",
            "subtlety",
            "dataset",
            "image_series_uid",
            "roi_mask_series_uid",
            "cropped_image_series_uid",
        ]
        return dfv.T

    def _run_validity_summary(self, dfv: pd.DataFrame) -> pd.DataFrame:
        """Summarizes the validity assessment"""
        n = dfv["N"].sum()
        nv = dfv["Valid"].sum()
        pv = nv / n
        dv = {"N": n, "Valid": nv, "Validity": pv}
        dfv = pd.DataFrame(data=dv, index=[0]).T
        dfv.columns = ["Values"]
        return dfv

    def _validate_mass_shape(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_nonnull(s=self._df["mass_shape"])
        pv = self._p_values_nonnull(s=self._df["mass_shape"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def _validate_mass_margins(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_nonnull(s=self._df["mass_margins"])
        pv = self._p_values_nonnull(s=self._df["mass_margins"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)
