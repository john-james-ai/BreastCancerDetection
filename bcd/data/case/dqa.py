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
# Modified   : Friday September 22nd 2023 08:12:24 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Data Quality Module"""
import sys
import os
import logging
from abc import abstractmethod

import pandas as pd

from bcd.data.base import DQA, DQAResult

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class CaseDQA(DQA):
    @property
    def patient_id_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_containing(s=self._df["patient_id"], substr="P_")
        pv = self._p_values_containing(s=self._df["patient_id"], substr="P_")
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def breast_density_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(
            s=self._df["breast_density"].astype("int32"), min_value=1, max_value=4
        )
        pv = self._p_within_range(
            s=self._df["breast_density"].astype("int32"), min_value=1, max_value=4
        )
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def left_or_right_breast_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["left_or_right_breast"], values=["LEFT", "RIGHT"])
        pv = self._p_values_valid(s=self._df["left_or_right_breast"], values=["LEFT", "RIGHT"])
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
    def abnormality_id_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(
            s=self._df["abnormality_id"].astype("int32"), min_value=1, max_value=10
        )
        pv = self._p_within_range(
            s=self._df["abnormality_id"].astype("int32"), min_value=1, max_value=10
        )
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def abnormality_type_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["abnormality_type"], values=["calcification", "mass"])
        pv = self._p_values_valid(s=self._df["abnormality_type"], values=["calcification", "mass"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def assessment_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(
            s=self._df["assessment"].astype("int32"), min_value=0, max_value=6
        )
        pv = self._p_within_range(
            s=self._df["assessment"].astype("int32"), min_value=0, max_value=6
        )
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def pathology_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["pathology"], values=["BENIGN", "MALIGNANT"])
        pv = self._p_values_valid(s=self._df["pathology"], values=["BENIGN", "MALIGNANT"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def subtlety_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_within_range(s=self._df["subtlety"].astype("int32"), min_value=1, max_value=5)
        pv = self._p_within_range(s=self._df["subtlety"].astype("int32"), min_value=1, max_value=5)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def dataset_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_valid(s=self._df["dataset"], values=["train", "test"])
        pv = self._p_values_valid(s=self._df["dataset"], values=["train", "test"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def image_series_uid_validity(self) -> pd.Series:
        sop = "1.3.6.1.4.1.9590"
        n = self._n_rows(df=self._df)
        nv = self._n_values_containing(s=self._df["image_series_uid"], substr=sop)
        pv = self._p_values_containing(s=self._df["image_series_uid"], substr=sop)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def roi_mask_series_uid_validity(self) -> pd.Series:
        sop = "1.3.6.1.4.1.9590"
        n = self._n_rows(df=self._df)
        nv = self._n_values_containing(s=self._df["roi_mask_series_uid"], substr=sop)
        pv = self._p_values_containing(s=self._df["roi_mask_series_uid"], substr=sop)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def cropped_image_series_uid_validity(self) -> pd.Series:
        sop = "1.3.6.1.4.1.9590"
        n = self._n_rows(df=self._df)
        nv = self._n_values_containing(s=self._df["cropped_image_series_uid"], substr=sop)
        pv = self._p_values_containing(s=self._df["cropped_image_series_uid"], substr=sop)
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @abstractmethod
    def validate(self) -> pd.DataFrame:
        "Validates the data and returns a boolean mask of cell validity."

    def validate_patient_id(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        return self._df["patient_id"].str.contains("P_", regex=False)

    def validate_breast_density(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        return self._df["breast_density"].astype("int32").between(left=1, right=4)

    def validate_side(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        values = ["RIGHT", "LEFT"]
        return self._df["left_or_right_breast"].isin(values)

    def validate_image_view(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        values = ["CC", "MLO"]
        return self._df["image_view"].isin(values)

    def validate_abnormality_id(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        return self._df["abnormality_id"].between(left=1, right=10)

    def validate_abnormality_type(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        values = ["calcification", "mass"]
        return self._df["abnormality_type"].isin(values)

    def validate_assessment(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        return self._df["assessment"].astype("int32").between(left=1, right=6)

    def validate_pathology(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        values = ["BENIGN", "MALIGNANT"]
        return self._df["pathology"].isin(values)

    def validate_subtlety(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        return self._df["subtlety"].astype("int32").between(left=1, right=5)

    def validate_dataset(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        values = ["train", "test"]
        return self._df["dataset"].isin(values)

    def validate_series_uid(self, uid: str) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable.

        Args:
            uid (str): The series uid column name
        """
        substr = "1.3.6.1.4.1.9590"
        return self._df[uid].str.contains(substr, regex=False)


# ------------------------------------------------------------------------------------------------ #
class CalcCaseDQA(CaseDQA):
    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        self._df = pd.read_csv(self._filepath)
        self._completeness = None
        self._uniqueness = None
        self._validity = None

    @property
    def calc_type_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_nonnull(s=self._df["calc_type"])
        pv = self._p_values_nonnull(s=self._df["calc_type"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def calc_distribution_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_nonnull(s=self._df["calc_distribution"])
        pv = self._p_values_nonnull(s=self._df["calc_distribution"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def validate(self) -> pd.DataFrame:
        "Validates the data and returns a boolean mask of cell validity."
        pid = self.validate_patient_id()
        bd = self.validate_breast_density()
        side = self.validate_side()
        view = self.validate_image_view()
        aid = self.validate_abnormality_id()
        at = self.validate_abnormality_type()
        asmt = self.validate_assessment()
        ct = self.validate_calcification_type()
        cd = self.validate_calcification_distribution()
        path = self.validate_pathology()
        sub = self.validate_subtlety()
        ds = self.validate_dataset()
        isuid = self.validate_series_uid(uid="image_series_uid")
        roisuid = self.validate_series_uid(uid="roi_mask_series_uid")
        cropsuid = self.validate_series_uid(uid="cropped_image_series_uid")
        dfv = pd.concat(
            [pid, bd, side, view, aid, at, ct, cd, asmt, path, sub, ds, isuid, roisuid, cropsuid],
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
        return dfv

    def validate_calcification_type(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        return self._df["calcification_type"].notnull()

    def validate_calcification_distribution(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        return self._df["calcification_distribution"].notnull()

    def analyze_validity(self) -> DQAResult:
        """Executes a Validity Assessment"""
        dvd = self._analyze_validity()
        dvs = self._summarize_validity(dfv=dvd)
        result = DQAResult(summary=dvs, detail=dvd)
        return result

    def _analyze_validity(self) -> pd.DataFrame:
        """Performs detailed validity assessment"""
        pid = self.patient_id_validity
        bd = self.breast_density_validity
        side = self.left_or_right_breast_validity
        view = self.image_view_validity
        aid = self.abnormality_id_validity
        at = self.abnormality_type_validity
        ct = self.calc_type_validity
        cd = self.calc_distribution_validity
        asmt = self.assessment_validity
        path = self.pathology_validity
        sbty = self.subtlety_validity
        ds = self.dataset_validity
        img_suid = self.image_series_uid_validity
        roi_suid = self.roi_mask_series_uid_validity
        crop_suid = self.cropped_image_series_uid_validity
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

    def _summarize_validity(self, dfv: pd.DataFrame) -> pd.DataFrame:
        """Summarizes the validity assessment"""
        n = dfv["N"].sum()
        nv = dfv["Valid"].sum()
        pv = nv / n
        dv = {"N": n, "Valid": nv, "Validity": pv}
        dfv = pd.DataFrame(data=dv, index=[0]).T
        dfv.columns = ["Values"]
        return dfv


# ------------------------------------------------------------------------------------------------ #
class MassCaseDQA(CaseDQA):
    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        self._df = pd.read_csv(self._filepath)

    @property
    def mass_shape_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_nonnull(s=self._df["mass_shape"])
        pv = self._p_values_nonnull(s=self._df["mass_shape"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    @property
    def mass_margins_validity(self) -> pd.Series:
        n = self._n_rows(df=self._df)
        nv = self._n_values_nonnull(s=self._df["mass_margins"])
        pv = self._p_values_nonnull(s=self._df["mass_margins"])
        dv = {"N": n, "Valid": nv, "Validity": pv}
        return pd.Series(dv)

    def validate(self) -> pd.DataFrame:
        "Validates the data and returns a boolean mask of cell validity."
        pid = self.validate_patient_id()
        bd = self.validate_breast_density()
        side = self.validate_side()
        view = self.validate_image_view()
        aid = self.validate_abnormality_id()
        at = self.validate_abnormality_type()
        ms = self.validate_mass_shape()
        mm = self.validate_mass_margins()
        asmt = self.validate_assessment()
        path = self.validate_pathology()
        sub = self.validate_subtlety()
        ds = self.validate_dataset()
        isuid = self.validate_series_uid(uid="image_series_uid")
        roisuid = self.validate_series_uid(uid="roi_mask_series_uid")
        cropsuid = self.validate_series_uid(uid="cropped_image_series_uid")
        dfv = pd.concat(
            [pid, bd, side, view, aid, at, ms, mm, asmt, path, sub, ds, isuid, roisuid, cropsuid],
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
        return dfv

    def validate_mass_shape(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        return self._df["mass_shape"].notnull()

    def validate_mass_margins(self) -> pd.Series:
        """Returns a boolean mask indicating validity of the variable."""
        return self._df["mass_margins"].notnull()

    def analyze_validity(self) -> DQAResult:
        """Executes a Validity Assessment"""
        dvd = self._analyze_validity()
        dvs = self._summarize_validity(dfv=dvd)
        result = DQAResult(summary=dvs, detail=dvd)
        return result

    def _analyze_validity(self) -> pd.DataFrame:
        """Performs detailed validity assessment"""
        pid = self.patient_id_validity
        bd = self.breast_density_validity
        side = self.left_or_right_breast_validity
        view = self.image_view_validity
        aid = self.abnormality_id_validity
        at = self.abnormality_type_validity
        ms = self.mass_shape_validity
        mm = self.mass_margins_validity
        asmt = self.assessment_validity
        path = self.pathology_validity
        sbty = self.subtlety_validity
        ds = self.dataset_validity
        img_suid = self.image_series_uid_validity
        roi_suid = self.roi_mask_series_uid_validity
        crop_suid = self.cropped_image_series_uid_validity
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

    def _summarize_validity(self, dfv: pd.DataFrame) -> pd.DataFrame:
        """Summarizes the validity assessment"""
        n = dfv["N"].sum()
        nv = dfv["Valid"].sum()
        pv = nv / n
        dv = {"N": n, "Valid": nv, "Validity": pv}
        dfv = pd.DataFrame(data=dv, index=[0]).T
        dfv.columns = ["Values"]
        return dfv
