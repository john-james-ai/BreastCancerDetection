#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/metadata/case.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:38 am                                              #
# Modified   : Saturday October 28th 2023 02:23:59 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Data Prep Module"""
from __future__ import annotations
import os
from typing import Union

import pandas as pd
import numpy as np
from studioai.preprocessing.encode import RankFrequencyEncoder
# pylint: disable=unused-import
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from bcd.preprocess.metadata.base import DataPrep


# ------------------------------------------------------------------------------------------------ #
class CasePrep(DataPrep):
    """Performs Case metadata preparation."""
    # pylint: disable=arguments-differ

    def prep(
        self,
        calc_train_fp: str,
        calc_test_fp: str,
        mass_train_fp: str,
        mass_test_fp: str,
        case_fp: str,
        case_series_fp: str,
        force: bool = False,
        result: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """Combines training and test cases into a single csv case file.

        Args:
            calc_train_fp, calc_test_fp, mass_train_fp, mass_test_fp (str): The file paths to the
                calcification and mass training and test sets.
            calc_fp, mass_fp (str): Path to output calcification and mass datasets.
            case_series_fp (str): File path to the case/series cross-reference file.
            force (bool): Whether to force execution if output already exists. Default is False.
            result (bool): Whether the result should be returned. Default is False.

        Returns
            If result is True, the following tuple is returned: (df_cases,  df_case_series)
        """
        case_fp = os.path.abspath(case_fp)
        case_series_fp = os.path.abspath(case_series_fp)

        os.makedirs(os.path.dirname(case_fp), exist_ok=True)
        os.makedirs(os.path.dirname(case_series_fp), exist_ok=True)

        if force or not os.path.exists(case_fp):
            # Merge all case data into a single DataFrame
            df_cases = self._merge_cases(
                calc_train_fp=calc_train_fp,
                calc_test_fp=calc_test_fp,
                mass_train_fp=mass_train_fp,
                mass_test_fp=mass_test_fp,
            )

            df_cases = self._assign_case_id(df=df_cases)

            # Transform 'BENIGN WITHOUT CALLBACK' to 'BENIGN'
            df_cases["cancer"] = np.where(df_cases["pathology"] == "MALIGNANT", True, False)

            # Create the series/ case cross-reference file
            df_cases, df_case_series = self._create_case_series_xref(df=df_cases)

            # Save datasets
            df_cases.to_csv(case_fp, index=False)
            df_case_series.to_csv(case_series_fp, index=False)

        if result:
            df_cases = pd.read_csv(case_fp)
            df_case_series = pd.read_csv(case_series_fp)
            return (df_cases, df_case_series)

    def _merge_cases(
        self,
        calc_train_fp: str,
        calc_test_fp: str,
        mass_train_fp: str,
        mass_test_fp: str,
    ) -> pd.DataFrame:
        """Combines mass and calcification train and test files into a single file."""
        calc_train_fp = os.path.abspath(calc_train_fp)
        calc_test_fp = os.path.abspath(calc_test_fp)
        mass_train_fp = os.path.abspath(mass_train_fp)
        mass_test_fp = os.path.abspath(mass_test_fp)

        df_calc_train = pd.read_csv(calc_train_fp)
        df_calc_test = pd.read_csv(calc_test_fp)
        df_mass_train = pd.read_csv(mass_train_fp)
        df_mass_test = pd.read_csv(mass_test_fp)

        df_calc_train["fileset"] = "train"
        df_calc_test["fileset"] = "test"
        df_mass_train["fileset"] = "train"
        df_mass_test["fileset"] = "test"

        df_calc_train = self._format_column_names(df=df_calc_train)
        df_calc_test = self._format_column_names(df=df_calc_test)
        df_mass_train = self._format_column_names(df=df_mass_train)
        df_mass_test = self._format_column_names(df=df_mass_test)

        df = pd.concat([df_calc_train, df_calc_test, df_mass_train, df_mass_test], axis=0)
        df.loc[df["abnormality_type"] == "mass", "calc_type"] = "NOT APPLICABLE"
        df.loc[df["abnormality_type"] == "mass", "calc_distribution"] = "NOT APPLICABLE"
        df.loc[df["abnormality_type"] == "calcification", "mass_shape"] = "NOT APPLICABLE"
        df.loc[df["abnormality_type"] == "calcification", "mass_margins"] = "NOT APPLICABLE"
        return df

    def _assign_case_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign a case id to each observation."""
        df["case_id"] = (
            df["abnormality_type"].apply(lambda x: x.capitalize())
            + "-"
            + df["fileset"].apply(lambda x: x.capitalize())
            + "_"
            + df["patient_id"]
            + "_"
            + df["left_or_right_breast"]
            + "_"
            + df["image_view"]
            + "_"
            + df["abnormality_id"].astype("str")
        )

        return df

    def _create_case_series_xref(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Creates a case series cross-reference"""

        def extract_series_uid(filepath) -> str:
            return filepath.split("/")[2]

        csxi = df[["case_id", "image_file_path"]].copy()
        csxi["series_uid"] = csxi["image_file_path"].apply(extract_series_uid)
        csxi["series_description"] = "full mammogram images"
        csxi = csxi.drop(columns=["image_file_path"])

        csxr = df[["case_id", "ROI_mask_file_path"]].copy()
        csxr["series_uid"] = csxr["ROI_mask_file_path"].apply(extract_series_uid)
        csxr["series_description"] = "ROI mask images"
        csxr = csxr.drop(columns=["ROI_mask_file_path"])

        csxc = df[["case_id", "cropped_image_file_path"]].copy()
        csxc["series_uid"] = csxc["cropped_image_file_path"].apply(extract_series_uid)
        csxc["series_description"] = "cropped images"
        csxc = csxc.drop(columns=["cropped_image_file_path"])

        df_case_series = pd.concat([csxi, csxr, csxc], axis=0)
        df_case_series = df_case_series.drop_duplicates(subset=["series_uid", "case_id"])
        df_cases = df.drop(
            columns=["image_file_path", "ROI_mask_file_path", "cropped_image_file_path"]
        )

        return df_cases, df_case_series


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
        self, max_iter: int = 50, initial_strategy: str = "most_frequent", random_state: int = None
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
CALC_DISTRIBUTIONS = ["CLUSTERED", "LINEAR", "REGIONAL", "DIFFUSELY_SCATTERED", "SEGMENTAL"]
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
MASS_MARGINS = ["SPICULATED", "ILL_DEFINED", "CIRCUMSCRIBED", "OBSCURED", "MICROLOBULATED"]

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
                df = self._encode_column(df=df, prefix=data["prefix"], col=feature, value=value)
        return df

    def _encode_column(self, df, prefix, col, value):
        newcol = prefix + "_" + value
        df[newcol] = np.where(df[col].str.contains(value), 1, 0)
        return df

    def _save(self, df: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(self._destination_fp), exist_ok=True)
        df.to_csv(self._destination_fp, index=False)
