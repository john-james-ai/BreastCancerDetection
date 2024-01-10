#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/analyze/dqa/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday September 23rd 2023 12:45:12 am                                            #
# Modified   : Tuesday January 2nd 2024 05:46:58 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Package base module for Data Quality Analysis"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import pandas as pd

from bcd import DataClass


# ================================================================================================ #
#                                      DATA QUALITY                                                #
# ================================================================================================ #
@dataclass
class DQAResult:
    """Encapsulates the result of a data quality analysis"""

    summary: DQASummary
    detail: pd.DataFrame


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DQASummary(DataClass):
    """Base class for data quality analysis result summary."""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Completeness(DQASummary):
    """Encapsulates a summary of data completeness."""

    dataset: str
    records: int
    complete_records: int
    record_completeness: float
    data_values: int
    complete_data_values: int
    data_value_completeness: float


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Uniqueness(DQASummary):
    """Encapsulates a summary of data uniqueness"""

    dataset: str
    records: int
    unique_records: int
    record_uniqueness: float
    data_values: int
    unique_data_values: int
    data_value_uniqueness: float


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Validity(DQASummary):
    """Encapsulates a summary of data validity"""

    dataset: str
    records: int
    valid_records: int
    record_validity: float
    data_values: int
    valid_data_values: int
    data_value_validity: float


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Consistency(DQASummary):
    """Encapsulates a summary of data consistency"""

    dataset: str
    records: int
    consistent_records: int
    record_consistency: float
    data_values: int
    consistent_data_values: int
    data_value_consistency: float


# ------------------------------------------------------------------------------------------------ #
class DQA(ABC):
    """Data Quality Analysis Base Class"""

    def __init__(self, data: pd.DataFrame, name: str) -> None:
        self._df = data
        self._name = name

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
        dc["Missing"] = dc["N"] - dc["Complete"]
        dc["Completeness"] = dc["Complete"] / dc["N"]
        dc = dc[["N", "Complete", "Missing", "Completeness"]]

        # Summary completeness
        nrows = self._df.shape[0]
        ncr = mask.all(axis=1).sum()
        pcr = round(ncr / nrows, 3)

        ncells = self._df.shape[0] * self._df.shape[1]
        ncc = mask.sum().sum()
        pcc = round(ncc / ncells, 3)

        sc = Completeness(
            dataset=self._name,
            records=nrows,
            complete_records=ncr,
            record_completeness=pcr,
            data_values=ncells,
            complete_data_values=ncc,
            data_value_completeness=pcc,
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
        du["Duplicate"] = du["N"] - du["Unique"]
        du["Uniqueness"] = du["Unique"] / du["N"]
        du = du[["N", "Unique", "Duplicate", "Uniqueness"]]

        # Summary Uniqueness
        nrows = self._df.shape[0]
        nur = len(self._df.drop_duplicates())
        pur = round(nur / nrows, 3)

        ncells = ncells = self._df.shape[0] * self._df.shape[1]
        nuc = du["Unique"].sum()
        puc = round(nuc / ncells, 3)

        su = Uniqueness(
            dataset=self._name,
            records=nrows,
            unique_records=nur,
            record_uniqueness=pur,
            data_values=ncells,
            unique_data_values=nuc,
            data_value_uniqueness=puc,
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
        dv["Invalid"] = dv["N"] - dv["Valid"]
        dv["Validity"] = dv["Valid"] / dv["N"]

        dv = dv[["N", "Valid", "Invalid", "Validity"]]

        # Summary Validity
        nrows = self._df.shape[0]
        nvr = validation_mask.all(axis=1).sum()
        pvr = round(nvr / nrows, 3)

        ncells = self._df.shape[0] * self._df.shape[1]
        nvc = ncells - validation_mask.eq(0).sum().sum()
        pvc = round(nvc / ncells, 3)

        vs = Validity(
            dataset=self._name,
            records=nrows,
            valid_records=nvr,
            record_validity=pvr,
            data_values=ncells,
            valid_data_values=nvc,
            data_value_validity=pvc,
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

    def get_invalid_data(self, subset: Union[list, str] = None) -> pd.DataFrame:
        """Returns dataframe of rows containing invalid non-null data."""
        validation_mask = self.validate()
        if subset is None:
            return self._df[~validation_mask.values.all(axis=1)]
        else:
            validation_mask = validation_mask[subset]
            return self._df.loc[~validation_mask.values]
