#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data_prep/case.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:38 am                                              #
# Modified   : Sunday December 31st 2023 09:59:16 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Data Prep Module"""
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=unused-import, disable=arguments-differ
# ------------------------------------------------------------------------------------------------ #
from __future__ import annotations

import os
from typing import Union

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from studioai.preprocessing.encode import RankFrequencyEncoder

from bcd.data_prep.base import DataPrep


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=unused-import, disable=arguments-differ
# ------------------------------------------------------------------------------------------------ #
class CasePrep(DataPrep):
    """Performs Case metadata preparation."""

    def prep(
        self,
        calc_train_fp: str,
        calc_test_fp: str,
        mass_train_fp: str,
        mass_test_fp: str,
        case_fp: str,
        force: bool = False,
        result: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """Combines training and test cases into a single csv case file.

        Args:
            calc_train_fp, calc_test_fp, mass_train_fp, mass_test_fp (str): The file paths to the
                calcification and mass training and test sets.
            case_fp (str): Path to output calcification and mass datasets.

            force (bool): Whether to force execution if output already exists. Default is False.
            result (bool): Whether the result should be returned. Default is False.

        Returns
            If result is True, the case dataframe is returned.
        """
        case_fp = os.path.abspath(case_fp)

        os.makedirs(os.path.dirname(case_fp), exist_ok=True)

        if force or not os.path.exists(case_fp):
            # Merge all case data into a single DataFrame
            df_cases = self._merge_cases(
                calc_train_fp=calc_train_fp,
                calc_test_fp=calc_test_fp,
                mass_train_fp=mass_train_fp,
                mass_test_fp=mass_test_fp,
            )

            # Set morphological features to NA as appropriate
            df_cases.loc[
                df_cases["abnormality_type"] == "mass", "calc_type"
            ] = "NOT APPLICABLE"
            df_cases.loc[
                df_cases["abnormality_type"] == "mass", "calc_distribution"
            ] = "NOT APPLICABLE"
            df_cases.loc[
                df_cases["abnormality_type"] == "calcification", "mass_shape"
            ] = "NOT APPLICABLE"
            df_cases.loc[
                df_cases["abnormality_type"] == "calcification", "mass_margins"
            ] = "NOT APPLICABLE"

            # Assign the mammogram id.
            df_cases = self._assign_mmg_id(df=df_cases)

            # Transform 'BENIGN WITHOUT CALLBACK' to 'BENIGN'
            df_cases["cancer"] = np.where(
                df_cases["pathology"] == "MALIGNANT", True, False
            )

            # Drop the filename columns.
            columns_to_drop = [
                "image_file_path",
                "cropped_image_file_path",
                "ROI_mask_file_path",
            ]
            df_cases = df_cases.drop(columns=columns_to_drop)

            # Save datasets
            df_cases.to_csv(case_fp, index=False)

        if result:
            return pd.read_csv(case_fp)

    def _merge_cases(
        self,
        calc_train_fp: str,
        calc_test_fp: str,
        mass_train_fp: str,
        mass_test_fp: str,
    ) -> pd.DataFrame:
        """Combines mass and calcification train and test files into a single file."""
        # Extracts absolute paths, a pre-emptive measure in case
        # jupyter book can't access the path
        calc_train_fp = os.path.abspath(calc_train_fp)
        calc_test_fp = os.path.abspath(calc_test_fp)
        mass_train_fp = os.path.abspath(mass_train_fp)
        mass_test_fp = os.path.abspath(mass_test_fp)

        df_calc_train = pd.read_csv(calc_train_fp)
        df_calc_test = pd.read_csv(calc_test_fp)
        df_mass_train = pd.read_csv(mass_train_fp)
        df_mass_test = pd.read_csv(mass_test_fp)

        # Add the filesets so that we can distinguish training
        # and test data
        df_calc_train["fileset"] = "training"
        df_calc_test["fileset"] = "test"
        df_mass_train["fileset"] = "training"
        df_mass_test["fileset"] = "test"

        # Replace spaces in column names with underscores.
        df_calc_train = self._format_column_names(df=df_calc_train)
        df_calc_test = self._format_column_names(df=df_calc_test)
        df_mass_train = self._format_column_names(df=df_mass_train)
        df_mass_test = self._format_column_names(df=df_mass_test)

        # Concatenate the files
        df = pd.concat(
            [df_calc_train, df_calc_test, df_mass_train, df_mass_test], axis=0
        )

        return df

    def _assign_mmg_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign a mammogram id to each observation."""
        df["mmg_id"] = (
            df["abnormality_type"].apply(lambda x: x[0:4].capitalize())
            + "-"
            + df["fileset"].apply(lambda x: x.capitalize())
            + "_"
            + df["patient_id"]
            + "_"
            + df["left_or_right_breast"].apply(lambda x: x.upper())
            + "_"
            + df["image_view"].apply(lambda x: x.upper())
        )

        return df

    def _format_column_names(self, df: pd.DataFrame) -> str:
        """Replaces spaces in column names with underscores."""

        def replace_columns(colname: str) -> str:
            return colname.replace(" ", "_")

        df.columns = df.columns.to_series().apply(replace_columns)
        return df


# ------------------------------------------------------------------------------------------------ #
class CaseImputer:
    """Imputes the missing values in the case dataset using Multiple Imputation by Chained Equations

    Args:
        max_iter (int): Maximum number of imputation rounds to perform before returning
        the imputations computed during the final round.
        initial_strategy (str): Which strategy to use to initialize the missing values.
            Valid values include: {'mean', 'median', 'most_frequent', 'constant'},
            default=most_frequent'
        random_state (int): The seed of the pseudo random number generator to use.

    """

    def __init__(
        self,
        max_iter: int = 50,
        initial_strategy: str = "most_frequent",
        random_state: int = None,
    ) -> None:
        self._max_iter = max_iter
        self._initial_strategy = initial_strategy
        self._random_state = random_state
        self._encoded_values = {}
        self._dtypes = None
        self._enc = None
        self._imp = None

    def fit(self, df: pd.DataFrame) -> CaseImputer:
        """Fits the data to the imputer

        Instantiates the encoder, encodes the data and creates a
        map of columns to valid encoded values. We capture these
        values in order to map imputed values
        back to valid values before we inverse transform.

        Args:
            df (pd.DataFrame): Imputed DataFrame
        """
        self._dtypes = df.dtypes.astype(str).replace("0", "object").to_dict()
        self._enc = RankFrequencyEncoder()
        df_enc = self._enc.fit_transform(df=df)
        self._extract_encoded_values(df=df_enc)

        # Get complete cases for imputer training (fit)
        df_enc_complete = df_enc.dropna(axis=0)

        self._imp = IterativeImputer(
            max_iter=self._max_iter,
            initial_strategy=self._initial_strategy,
            random_state=self._random_state,
        )
        self._imp.fit(X=df_enc_complete.values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs the imputation and returns the imputed DataFrame

        Args:
            df (pd.DataFrame): Imputed DataFrame

        """
        df_enc = self._enc.transform(df=df)
        imp = self._imp.transform(X=df_enc.values)
        df_imp = pd.DataFrame(data=imp, columns=df.columns)
        df_imp = self._map_imputed_values(df=df_imp)
        df_imp = self._enc.inverse_transform(df=df_imp)
        df_imp = df_imp.astype(self._dtypes)
        return df_imp

    def _extract_encoded_values(self, df: pd.DataFrame) -> None:
        """Creates a dictionary of valid values by column."""
        for col in df.columns:
            valid = df[col].dropna()
            self._encoded_values[col] = valid.unique()

    def _map_imputed_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps values to valid values (used after imputation)"""
        # pylint: disable=cell-var-from-loop
        for col in df.columns:
            values = np.array(sorted(self._encoded_values[col]))
            df[col] = df[col].apply(lambda x: values[np.argmin(np.abs(x - values))])
        return df


# ------------------------------------------------------------------------------------------------ #
#                                   CASE TRANSFORMER                                               #
# ------------------------------------------------------------------------------------------------ #
CALC_TYPES = [
    "AMORPHOUS",
    "COARSE",
    "DYSTROPHIC",
    "EGGSHELL",
    "FINE_LINEAR_BRANCHING",
    "LARGE_RODLIKE",
    "LUCENT_CENTERED",
    "MILK_OF_CALCIUM",
    "PLEOMORPHIC",
    "PUNCTATE",
    "ROUND_AND_REGULAR",
    "SKIN",
    "VASCULAR",
]
CALC_DISTRIBUTIONS = [
    "CLUSTERED",
    "LINEAR",
    "REGIONAL",
    "DIFFUSELY_SCATTERED",
    "SEGMENTAL",
]
MASS_SHAPES = [
    "IRREGULAR",
    "ARCHITECTURAL_DISTORTION",
    "OVAL",
    "LYMPH_NODE",
    "LOBULATED",
    "FOCAL_ASYMMETRIC_DENSITY",
    "ROUND",
    "ASYMMETRIC_BREAST_TISSUE",
]
MASS_MARGINS = [
    "SPICULATED",
    "ILL_DEFINED",
    "CIRCUMSCRIBED",
    "OBSCURED",
    "MICROLOBULATED",
]

ENC_VARS = {
    "abnormality_type": {"prefix": "AT", "values": ["calcification", "mass"]},
    "left_or_right_breast": {"prefix": "LR", "values": ["LEFT", "RIGHT"]},
    "image_view": {"prefix": "IV", "values": ["CC", "MLO"]},
    "calc_type": {"prefix": "CT", "values": CALC_TYPES},
    "calc_distribution": {"prefix": "CD", "values": CALC_DISTRIBUTIONS},
    "mass_shape": {"prefix": "MS", "values": MASS_SHAPES},
    "mass_margins": {"prefix": "MM", "values": MASS_MARGINS},
}


# ------------------------------------------------------------------------------------------------ #
class CaseTransformer:
    """Collapses morphological categories and dummy encodes nominal variables.

    The CBIS-DDSM has 45 calcification types, 9 calcification distributions, 20 mass shapes, and
    19 mass margins, many of which are compound categories, in that two or more categories are
    combined. For instance, calcification type 'ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS' indicates
    three different types: 'ROUND_AND_REGULAR', 'PUNCTATE', and 'AMORPHOUS'. Segregating these
    compound categories into separate categories will drastically reduce the number of categories
    to analyze. More importantly, it aligns our data and the analyses with the common morphological
    taxonomy. So, task one is to extract the unary morphological categories from the
    compound classifications.

    Args:
        source_fp (str): Path to source file
        destination_fp (str): Path to destination file
    """

    def __init__(self, source_fp: str, destination_fp: str) -> None:
        self._source_fp = os.path.abspath(source_fp)
        self._destination_fp = os.path.abspath(destination_fp)

    def transform(self) -> pd.DataFrame:
        df = pd.read_csv(self._source_fp)
        df["cancer"] = np.where(df["cancer"] == True, 1, 0)  # noqa
        df_enc = self._encode_dataset(df=df)
        self._save(df=df_enc)
        return df_enc

    def _encode_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature, data in ENC_VARS.items():
            for value in data["values"]:
                df = self._encode_column(
                    df=df, prefix=data["prefix"], col=feature, value=value
                )
        return df

    def _encode_column(self, df, prefix, col, value):
        newcol = prefix + "_" + value
        df[newcol] = np.where(df[col].str.contains(value), 1, 0)
        return df

    def _save(self, df: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(self._destination_fp), exist_ok=True)
        df.to_csv(self._destination_fp, index=False)
