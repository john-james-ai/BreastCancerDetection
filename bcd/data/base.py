#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /bcd/data/base.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 31st 2023 07:36:47 pm                                               #
# Modified   : Friday September 22nd 2023 09:50:38 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# %%
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from studioai.data.dataset import Dataset


# ================================================================================================ #
class DataPrep(ABC):
    @abstractmethod
    def prep(self, force: bool = False, result: bool = False, *args, **kwargs) -> None:
        """Performs the data preparation task."""

    def _format_column_names(self, df: pd.DataFrame) -> str:
        """Replaces spaces in column names with underscores."""

        def replace_columns(colname: str) -> str:
            return colname.replace(" ", "_")

        df.columns = df.columns.to_series().apply(replace_columns)
        return df


# ================================================================================================ #
#                                      DATA QUALITY                                                #
# ================================================================================================ #
@dataclass
class DQAResult:
    summary: pd.DataFrame
    detail: pd.DataFrame


# ------------------------------------------------------------------------------------------------ #
class DQA(ABC):
    """Data Quality Analysis Base Class"""

    @abstractmethod
    def run(self) -> None:
        "Executes the data quality analysis"

    def run_completeness(self) -> DQAResult:
        """Executes the completeness analysis"""
        result = DQAResult(
            summary=self._run_completeness_summary(), detail=self._run_completeness_detail()
        )
        return result

    def _run_completeness_summary(self) -> pd.DataFrame:
        """Returns a summary completeness analysis"""
        nr = self._n_rows(df=self._df)
        ncr = self._n_rows_complete(df=self._df)
        pcr = self._p_rows_complete(df=self._df)

        nc = self._n(df=self._df)
        ncc = self._n_complete(df=self._df)
        pcc = self._p_complete(df=self._df)

        dc = {
            "Rows": nr,
            "Rows Complete": ncr,
            "Row Completeness": pcr,
            "Cells": nc,
            "Cells Complete": ncc,
            "Cell Completeness": pcc,
        }
        dfc = pd.DataFrame(data=dc, index=[0]).T
        dfc.columns = ["Values"]
        dfc = dfc.round(3)
        return dfc

    def _run_completeness_detail(self) -> pd.DataFrame:
        """Returns a detailed completeness analysis"""
        n = self._n_by_var(df=self._df)
        n_complete = self._n_complete_by_var(df=self._df)
        p_complete = self._p_complete_by_var(df=self._df)
        dfc = pd.concat([n, n_complete, p_complete], axis=1)
        dfc.columns = ["N", "Complete", "Completeness"]
        dfc = dfc.round(3)
        return dfc

    def run_uniqueness(self) -> DQAResult:
        """Executes the uniqueness analysis"""
        result = DQAResult(
            summary=self._run_uniqueness_summary(), detail=self._run_uniqueness_detail()
        )
        return result

    def _run_uniqueness_summary(self) -> pd.DataFrame:
        """Returns a summary uniqueness analysis"""
        nr = self._n_rows(df=self._df)
        nur = self._n_unique(df=self._df)
        pur = self._p_unique(df=self._df)

        du = {
            "Rows": nr,
            "Unique Rows": nur,
            "Row Uniqueness": pur,
        }
        dfu = pd.DataFrame(data=du, index=[0]).T
        dfu.columns = ["Values"]
        dfu = dfu.round(3)
        return dfu

    def _run_uniqueness_detail(self) -> pd.DataFrame:
        """Returns a detailed uniqueness analysis"""
        n = self._n_by_var(df=self._df)
        n_unique = self._n_unique_by_var(df=self._df)
        p_unique = self._p_unique_by_var(df=self._df)
        dfu = pd.concat([n, n_unique, p_unique], axis=1)
        dfu.columns = ["N", "Unique", "Uniqueness"]
        dfu = dfu.round(3)
        return dfu

    def _n(self, df: pd.DataFrame) -> int:
        """Returns number of cells in the dataframe."""
        return df.shape[0] * df.shape[1]

    def _n_complete(self, df: pd.DataFrame) -> int:
        """Returns number of complete cells in the DataFrame"""
        return df.notnull().sum().sum()

    def _p_complete(self, df: pd.DataFrame) -> float:
        """Returns proportion of cells complete in the DataFrame"""
        return self._n_complete(df=df) / self._n(df=df)

    def _n_by_var(self, df: pd.DataFrame) -> pd.Series:
        """Returns number of cells by variable."""
        return df.count() + df.isnull().sum()

    def _n_complete_by_var(self, df: pd.DataFrame) -> pd.Series:
        """Returns number of complete cells by variable"""
        return df.notnull().sum()

    def _p_complete_by_var(self, df: pd.DataFrame) -> pd.Series:
        """Returns proportion of complete cells by variable"""
        return self._n_complete_by_var(df=df) / self._n_by_var(df=df)

    def _n_rows(self, df: pd.DataFrame) -> int:
        """Returns number of rows in the dataframe."""
        return df.shape[0]

    def _n_rows_complete(self, df: pd.DataFrame) -> int:
        """Returns total number of rows complete for the dataset"""
        return len(df) - len(df[df.isnull()].any(axis=1))

    def _p_rows_complete(self, df: pd.DataFrame) -> float:
        """Returns proportion of complete rows for the dataset"""
        return self._n_rows_complete(df=df) / df.shape[0]

    def _n_unique(self, df: pd.DataFrame) -> int:
        """Returns the number of unique rows"""
        return len(df) + (len(df)) - len(df.drop_duplicates())

    def _p_unique(self, df: pd.DataFrame) -> float:
        """Returns the proportion of unique rows"""
        return self._n_unique(df=df) / len(df)

    def _n_unique_by_var(self, df: pd.DataFrame) -> pd.Series:
        """Returns the number of unique values by variable"""
        return df.nunique()

    def _p_unique_by_var(self, df: pd.DataFrame) -> pd.Series:
        """Returns the proportion of unique values by variable"""
        return self._n_unique_by_var(df=df) / self._n_by_var(df=df)

    def _n_range_valid(self, s: pd.Series, min_value: int, max_value: int) -> int:
        """Number of values that are between min and max value inclusive"""
        return s.between(left=min_value, right=max_value).sum()

    def _p_range_valid(self, s: pd.Series, min_value: int, max_value: int) -> float:
        """Returns proportion of values within range"""
        return self._n_range_valid(s=s, min_value=min_value, max_value=max_value) / len(s)

    def _n_values_valid(self, s: pd.Series, values: list) -> int:
        """Returns the number of values that are in list of valid values"""
        return s.isin(values).sum()

    def _p_values_valid(self, s: pd.Series, values: list) -> float:
        """Returns the proportion of values that are in list of valid values"""
        return self._n_values_valid(s=s, values=values) / len(s)

    def _n_values_containing(self, s: pd.Series, substr: str) -> int:
        """Returns the number of values containing the substring"""
        return s.str.contains(substr, regex=False).sum()

    def _p_values_containing(self, s: pd.Series, substr: str) -> float:
        """Returns the proportion of values containing the substring"""
        return s.str.contains(substr, regex=False).sum() / len(s)

    def _n_values_nonnull(self, s: pd.Series) -> int:
        """Returns the number of non-null values"""
        return s.notnull().sum()

    def _p_values_nonnull(self, s: pd.Series) -> float:
        """Returns the proportion of non-null values"""
        return self._n_values_nonnull(s=s) / len(s)


# ------------------------------------------------------------------------------------------------ #
class CBISDataset(Dataset):
    def summary(self) -> pd.DataFrame:
        counts = []
        cols = self._df.columns
        for col in cols:
            d = {}
            d[col] = self._df[col].value_counts()
            counts.append(d)
        df = pd.DataFrame(counts)
        return df
