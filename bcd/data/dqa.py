#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/dqa.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday September 23rd 2023 12:45:12 am                                            #
# Modified   : Saturday September 23rd 2023 04:18:44 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Package base module for Data Quality Analysis"""
from __future__ import annotations
import os
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import pandas as pd

from bcd.data.base import DataClass


# ================================================================================================ #
#                                      DATA QUALITY                                                #
# ================================================================================================ #
@dataclass
class DQAResult:
    summary: DQASummary
    detail: pd.DataFrame


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DQASummary(DataClass):
    """Base class for data quality analysis result summary."""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Completeness(DQASummary):
    dataset: str
    filename: str
    rows: int
    rows_complete: int
    row_completeness: float
    cells: int
    cells_complete: int
    cell_completeness: float


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Uniqueness(DQASummary):
    dataset: str
    filename: str
    rows: int
    unique_rows: int
    row_uniqueness: float
    cells: int
    unique_cells: int
    cell_uniqueness: float


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Validity(DQASummary):
    dataset: str
    filename: str
    rows: int
    rows_valid: int
    row_validity: float
    cells: int
    cells_valid: int
    cell_validity: float


# ------------------------------------------------------------------------------------------------ #
class DQA(ABC):
    """Data Quality Analysis Base Class"""

    def __init__(self, filepath: str, name: str = None) -> None:
        self._filepath = os.path.abspath(filepath)
        self._filename = os.path.basename(filepath)
        self._name = name or string.capwords(
            os.path.splitext(os.path.basename(self._filepath))[0].replace("_", " ")
        )

    @abstractmethod
    def validate(self) -> None:
        "Creates the validation mask."

    # -------------------------------------------------------------------------------------------- #
    def analyze_completeness(self) -> DQAResult:
        """Executes the completeness analysis"""
        mask = self._df.notnull()

        # Detail completeness
        dc = mask.sum(axis=0).to_frame()
        dc.columns = ["Complete"]
        dc["N"] = self._df.shape[0]
        dc["Completeness"] = dc["Complete"] / dc["N"]
        dc = dc[["N", "Complete", "Completeness"]]

        # Summary completeness
        nrows = self._df.shape[0]
        ncr = mask.all(axis=1).sum()
        pcr = round(ncr / nrows, 3)

        ncells = self._df.shape[0] * self._df.shape[1]
        ncc = mask.sum().sum()
        pcc = round(ncc / ncells, 3)

        sc = Completeness(
            dataset=self._name,
            filename=self._filename,
            rows=nrows,
            rows_complete=ncr,
            row_completeness=pcr,
            cells=ncells,
            cells_complete=ncc,
            cell_completeness=pcc,
        )

        # Format result
        result = DQAResult(summary=sc, detail=dc)
        return result

    # -------------------------------------------------------------------------------------------- #
    def analyze_uniqueness(self) -> DQAResult:
        """Executes the uniqueness analysis"""

        # Detailed Uniqueness
        du = self._df.nunique(axis=0).to_frame()
        du.columns = ["Unique"]
        du["N"] = self._df.shape[0]
        du["Uniqueness"] = du["Unique"] / du["N"]
        du = du[["N", "Unique", "Uniqueness"]]

        # Summary Uniqueness
        nrows = self._df.shape[0]
        nur = len(self._df.drop_duplicates())
        pur = round(nur / nrows, 3)

        ncells = ncells = self._df.shape[0] * self._df.shape[1]
        nuc = du["Unique"].sum()
        puc = round(nuc / ncells, 3)

        su = Uniqueness(
            dataset=self._name,
            filename=self._filename,
            rows=nrows,
            unique_rows=nur,
            row_uniqueness=pur,
            cells=ncells,
            unique_cells=nuc,
            cell_uniqueness=puc,
        )

        result = DQAResult(summary=su, detail=du)
        return result

    # -------------------------------------------------------------------------------------------- #
    def analyze_validity(self) -> DQAResult:
        """Executes a Validity Assessment"""
        validation_mask = self.validate()

        # Detailed Validity
        dv = validation_mask.sum(axis=0).to_frame()
        dv.columns = ["Valid"]
        dv["N"] = self._df.shape[0]
        dv["Validity"] = dv["Valid"] / dv["N"]
        dv = dv[["N", "Valid", "Validity"]]

        # Summary Validity
        nrows = self._df.shape[0]
        nvr = validation_mask.all(axis=1).sum()
        pvr = round(nvr / nrows, 3)

        ncells = self._df.shape[0] * self._df.shape[1]
        nvc = dv["Valid"].sum()
        pvc = round(nvc / ncells, 3)

        vs = Validity(
            dataset=self._name,
            filename=self._filename,
            rows=nrows,
            rows_valid=nvr,
            row_validity=pvr,
            cells=ncells,
            cells_valid=nvc,
            cell_validity=pvc,
        )

        # Format result
        result = DQAResult(summary=vs, detail=dv)
        return result

    # -------------------------------------------------------------------------------------------- #
    def get_complete_data(self) -> pd.DataFrame:
        """Returns complete rows"""
        return self._df.loc[self._df.notnull().all(axis=1)]

    def get_incomplete_data(self, subset: Union[list, str] = None) -> pd.DataFrame:
        """Returns rows of incomplete data

        Considering certain columns is optional.

        Args:
            subset (Union[str,list]): A column or list of columns to be evaluated
                for completeness.
        """
        if subset is None:
            return self._df.loc[self._df.isnull().any(axis=1)]
        elif isinstance(subset, str):
            return self._df.loc[self._df[subset].isnull()]
        else:
            return self._df.loc(self._df[subset].isnull().any(axis=1))

    # -------------------------------------------------------------------------------------------- #
    def get_unique_data(self) -> pd.DataFrame:
        """Returns unique rows"""
        return self._df.drop_duplicates()

    def get_duplicate_data(
        self, subset: Union[list, str] = None, keep: str = "first"
    ) -> pd.DataFrame:
        """Returns duplicate rows of data

        Considering certain columns is optional.

        Args:
            subset (Union[str,list]): A column or list of columns to be evaluated
                for duplication.
            keep (str): Either 'first', 'last', or false.  Determines which duplicates to mark.
                - first : Mark duplicates as True except for the first occurrence.
                - last : Mark duplicates as True except for the last occurrence.
                - False : Mark all duplicates as True.

        """
        return self._df.loc[self._df.duplicated(subset=subset, keep=keep)]

    # -------------------------------------------------------------------------------------------- #
    def get_valid_data(self) -> pd.DataFrame:
        """Returns dataframe of valid rows."""
        validation_mask = self.validate()
        return self._df[validation_mask.values.all(axis=1)]

    def get_invalid_data(self) -> pd.DataFrame:
        """Returns dataframe of rows containing invalid data."""
        validation_mask = self.validate()
        return self._df[~validation_mask.values.all(axis=1)]


# ------------------------------------------------------------------------------------------------ #
class Validator:
    """Encapsulates methods that evaluate data validity for data quality analysis."""

    def validate_patient_id(self, patient_id: pd.Series) -> pd.Series:
        """Evaluates validity of patient_id data

        Args:
            patient_id (pd.Series): A series containing patient identifiers.

        Returns: Boolean mask in Series format.

        """
        return patient_id.str.contains("P_", regex=False)

    def validate_series_uid(self, series_uid: pd.Series) -> pd.Series:
        """Evaluates validity of series_uid data

        Args:
            series_uid (pd.Series): A series containing series_uid data.

        Returns: Boolean mask in Series format.

        """
        substr = "1.3.6.1.4.1.9590"
        return series_uid.str.contains(substr, regex=False)

    def validate_breast_density(self, breast_density: pd.Series) -> pd.Series:
        """Evaluates validity of breast_density data using the BI-RADS standard.

        Args:
            breast_density (pd.Series): A series containing breast density data.

        Returns: Boolean mask in Series format.

        """
        return breast_density.astype("int32").between(left=1, right=4)

    def validate_side(self, side: pd.Series) -> pd.Series:
        """Evaluates validity of left or right side data.

        Args:
            side (pd.Series): A series containing left or right side breast data.

        Returns: Boolean mask in Series format.

        """
        values = ["RIGHT", "LEFT"]
        return side.isin(values)

    def validate_image_view(self, image_view: pd.Series) -> pd.Series:
        """Evaluates validity of image_view data.

        Valid values are CC (craniocaudal) and MLO (mediolateral oblique)

        Args:
            image_view (pd.Series): A series containing image view data.

        Returns: Boolean mask in Series format.

        """
        values = ["MLO", "CC"]
        return image_view.isin(values)

    def validate_between(self, data: pd.Series, left: int, right: int) -> pd.Series:
        """Evaluates whether the data values are within a given range.

        Args:
            data (pd.Series): A series numeric data.

        Returns: Boolean mask in Series format.

        """
        return data.astype("int32").between(left=left, right=right)

    def validate_abnormality_type(self, abnormality_type: pd.Series) -> pd.Series:
        """Evaluates validity of abnormality type

        Args:
            abnormality_type (pd.Series): Validates 'calcification', and 'mass' abnormality types

        Returns: Boolean mask in Series format.

        """
        values = ["calcification", "mass"]
        return abnormality_type.isin(values)

    def validate_assessment(self, assessment: pd.Series) -> pd.Series:
        """Evaluates validity of BI-RADS assessment values.

        Args:
            assessment (pd.Series): Validates BI-RADS assessment data

        Returns: Boolean mask in Series format.

        """
        return assessment.astype("int32").between(left=1, right=6)

    def validate_pathology(self, pathology: pd.Series) -> pd.Series:
        """Evaluates validity of pathology data.

        Args:
            pathology (pd.Series): Validates pathology is in ['BENIGN', 'MALIGNANT']

        Returns: Boolean mask in Series format.

        """
        values = ["BENIGN", "MALIGNANT"]
        return pathology.isin(values)

    def validate_subtlety(self, subtlety: pd.Series) -> pd.Series:
        """Evaluates validity of subtlety assessment values.

        Args:
            subtlety (pd.Series): Validates subtlety assessment data

        Returns: Boolean mask in Series format.

        """
        return subtlety.astype("int32").between(left=1, right=5)

    def validate_dataset(self, dataset: pd.Series) -> pd.Series:
        """Evaluates validity of dataset values

        Args:
            dataset (pd.Series): Validates dataset is in ['train', 'test']

        Returns: Boolean mask in Series format.

        """
        values = ["train", "test"]
        return dataset.isin(values)

    def validate_calc_type(self, calc_type: pd.Series) -> pd.Series:
        """Evaluates validity of calc_type values

        Args:
            calc_type (pd.Series): Calcification type values.

        Returns: Boolean mask in Series format.

        """
        values = [
            "AMORPHOUS",
            "PLEOMORPHIC",
            "ROUND_AND_REGULAR-LUCENT_CENTER-DYSTROPHIC",
            "PUNCTATE",
            "COARSE",
            "VASCULAR",
            "FINE_LINEAR_BRANCHING",
            "LARGE_RODLIKE",
            "PUNCTATE-LUCENT_CENTER",
            "VASCULAR-COARSE-LUCENT_CENTER-ROUND_AND_REGULAR-PUNCTATE",
            "ROUND_AND_REGULAR-EGGSHELL",
            "PUNCTATE-PLEOMORPHIC",
            "PLEOMORPHIC-FINE_LINEAR_BRANCHING",
            "DYSTROPHIC",
            "LUCENT_CENTER",
            "AMORPHOUS-PLEOMORPHIC",
            "ROUND_AND_REGULAR",
            "VASCULAR-COARSE-LUCENT_CENTERED",
            "COARSE-ROUND_AND_REGULAR",
            "COARSE-PLEOMORPHIC",
            "LUCENT_CENTERED",
            "VASCULAR-COARSE",
            "ROUND_AND_REGULAR-PUNCTATE",
            "ROUND_AND_REGULAR-LUCENT_CENTER",
            "COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED",
            "SKIN",
            "LUCENT_CENTER-PUNCTATE",
            "SKIN-PUNCTATE",
            "SKIN-PUNCTATE-ROUND_AND_REGULAR",
            "MILK_OF_CALCIUM",
            "PLEOMORPHIC-PLEOMORPHIC",
            "SKIN-COARSE-ROUND_AND_REGULAR",
            "AMORPHOUS-ROUND_AND_REGULAR",
            "ROUND_AND_REGULAR-PLEOMORPHIC",
            "ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS",
            "ROUND_AND_REGULAR-AMORPHOUS",
            "COARSE-ROUND_AND_REGULAR-LUCENT_CENTER",
            "LARGE_RODLIKE-ROUND_AND_REGULAR",
            "ROUND_AND_REGULAR-LUCENT_CENTER-PUNCTATE",
            "COARSE-LUCENT_CENTER",
            "PUNCTATE-AMORPHOUS",
            "ROUND_AND_REGULAR-LUCENT_CENTERED",
            "PUNCTATE-ROUND_AND_REGULAR",
            "EGGSHELL",
            "PUNCTATE-FINE_LINEAR_BRANCHING",
            "PLEOMORPHIC-AMORPHOUS",
            "PUNCTATE-AMORPHOUS-PLEOMORPHIC",
        ]

        return calc_type.isin(values)

    def validate_calc_distribution(self, calc_distribution: pd.Series) -> pd.Series:
        """Evaluates validity of calc_distribution values

        Args:
            calc_distribution (pd.Series): Calcification distribution values.

        Returns: Boolean mask in Series format.

        """
        values = [
            "CLUSTERED",
            "LINEAR",
            "REGIONAL",
            "DIFFUSELY_SCATTERED",
            "SEGMENTAL",
            "CLUSTERED-LINEAR",
            "CLUSTERED-SEGMENTAL",
            "LINEAR-SEGMENTAL",
            "REGIONAL-REGIONAL",
        ]
        return calc_distribution.isin(values)

    def validate_mass_shape(self, mass_shape: pd.Series) -> pd.Series:
        """Evaluates validity of mass_shape values

        Args:
            mass_shape (pd.Series): Mass shape values.

        Returns: Boolean mask in Series format.

        """
        values = [
            "IRREGULAR-ARCHITECTURAL_DISTORTION",
            "ARCHITECTURAL_DISTORTION",
            "OVAL",
            "IRREGULAR",
            "LYMPH_NODE",
            "LOBULATED-LYMPH_NODE",
            "LOBULATED",
            "FOCAL_ASYMMETRIC_DENSITY",
            "ROUND",
            "LOBULATED-ARCHITECTURAL_DISTORTION",
            "ASYMMETRIC_BREAST_TISSUE",
            "LOBULATED-IRREGULAR",
            "OVAL-LYMPH_NODE",
            "LOBULATED-OVAL",
            "ROUND-OVAL",
            "IRREGULAR-FOCAL_ASYMMETRIC_DENSITY",
            "ROUND-IRREGULAR-ARCHITECTURAL_DISTORTION",
            "ROUND-LOBULATED",
            "OVAL-LOBULATED",
            "IRREGULAR-ASYMMETRIC_BREAST_TISSUE",
        ]

        return mass_shape.isin(values)

    def validate_mass_margins(self, mass_margins: pd.Series) -> pd.Series:
        """Evaluates validity of mass_margins values

        Args:
            mass_margins (pd.Series): Mass margin values.

        Returns: Boolean mask in Series format.

        """
        values = [
            "SPICULATED",
            "ILL_DEFINED",
            "CIRCUMSCRIBED",
            "ILL_DEFINED-SPICULATED",
            "OBSCURED",
            "OBSCURED-ILL_DEFINED",
            "MICROLOBULATED",
            "MICROLOBULATED-ILL_DEFINED-SPICULATED",
            "MICROLOBULATED-SPICULATED",
            "CIRCUMSCRIBED-ILL_DEFINED",
            "MICROLOBULATED-ILL_DEFINED",
            "CIRCUMSCRIBED-OBSCURED",
            "OBSCURED-SPICULATED",
            "OBSCURED-ILL_DEFINED-SPICULATED",
            "CIRCUMSCRIBED-MICROLOBULATED",
            "OBSCURED-CIRCUMSCRIBED",
            "CIRCUMSCRIBED-SPICULATED",
            "CIRCUMSCRIBED-OBSCURED-ILL_DEFINED",
            "CIRCUMSCRIBED-MICROLOBULATED-ILL_DEFINED",
        ]

        return mass_margins.isin(values)

    def validate_series_description(self, series_description: pd.Series) -> pd.Series:
        """Evaluates validity of dataset values

        Args:
            series_description (pd.Series): Validates series description is in
                ["ROI mask images", "full mammogram images", "cropped images"]

        Returns: Boolean mask in Series format.

        """
        values = ["ROI mask images", "full mammogram images", "cropped images"]
        return series_description.isin(values)

    def validate_filepath(self, filepath: pd.Series) -> pd.Series:
        """Evaluates validity and existence of filepaths

        Args:
            filepaths (pd.Series): Validates filepaths

        Returns: Boolean mask in Series format.

        """
        return filepath.apply(lambda x: os.path.exists(x))

    def validate_image_bits(self, image_bits: pd.Series) -> pd.Series:
        """Evaluates validity of image_bits

        Args:
            dataset (pd.Series): Validates image_bits

        Returns: Boolean mask in Series format.

        """
        values = [8, 16]
        return image_bits.astype("int32").isin(values)
