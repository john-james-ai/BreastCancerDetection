#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data_prep/cbis.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Tuesday January 2nd 2024 08:43:06 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# pylint: disable=unused-import, arguments-renamed,arguments-differ
# ------------------------------------------------------------------------------------------------ #
"""DICOM Data Prep Module"""
from __future__ import annotations

import os

import dask
import numpy as np
import pandas as pd
import pydicom
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from studioai.preprocessing.encode import RankFrequencyEncoder

from bcd.dal.file import IOService
from bcd.data_prep.base import DataPrep
from bcd.utils.file import getsize
from bcd.utils.profile import profiler

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-renamed,arguments-differ


class CBISPrep(DataPrep):
    """Extracts DICOM data and integrates it with a single Case dataset staged for quality assessment.

    Iterates through the full mammography DICOM metadata in parallel, extracting image and pixel
    data and statistics, then combines the data with the case dataset.

    Args:
        case_filepath (str): Path to the case dataset.
        series_filepath (str): Path to the series dataset.
        cbis_filepath (str): Path to the combined case dicom dataset.
        force (bool): Whether to force execution if output already exists. Default is False.
    """

    def __init__(
        self,
        case_filepath: str,
        series_filepath: str,
        cbis_filepath: str,
        force: bool = False,
    ) -> None:
        super().__init__()
        self._case_filepath = os.path.abspath(case_filepath)
        self._series_filepath = os.path.abspath(series_filepath)
        self._cbis_filepath = os.path.abspath(cbis_filepath)
        self._force = force

    @profiler
    def prep(self) -> pd.DataFrame:
        """Extracts image metadata from the DICOM image files."""

        if self._force or not os.path.exists(self._cbis_filepath):
            # Reads the series metadata that contains subject, series, and
            # file location information
            cases = IOService.read(self._case_filepath)
            series = IOService.read(self._series_filepath)

            # Obtain the full mammogram images
            series = series.loc[series["series_description"] == "full mammogram images"]

            results = []
            # Graph of work is created and executed lazily at compute time.
            for _, study in series.iterrows():
                image_result = dask.delayed(self._extract_data)(study)
                results.append(image_result)

            # Compute the results and convert to dataframe
            results = dask.compute(*results)
            df = pd.DataFrame(data=results)

            # Merge the data with the case dataset
            df = cases.merge(df, on="mmg_id", how="inner")

            self._save(df=df, filepath=self._cbis_filepath)

            return df

        return IOService.read(self._cbis_filepath)

    def _extract_data(self, study: pd.Series) -> dict:
        """Reads study and dicom data from a file."""

        dcm = pydicom.dcmread(study["filepath"])
        img = dcm.pixel_array

        d = {}
        d["mmg_id"] = "_".join(study["subject_id"].split("_")[0:5])
        d["bit_depth"] = dcm.BitsStored
        d["rows"], d["cols"] = img.shape
        d["aspect_ratio"] = d["cols"] / d["rows"]
        d["size"] = d["rows"] * d["cols"]
        d["file_size"] = getsize(study["filepath"], as_bytes=True)
        d["min_pixel_value"] = dcm.SmallestImagePixelValue
        d["max_pixel_value"] = dcm.LargestImagePixelValue
        d["mean_pixel_value"] = np.mean(img)
        d["std_pixel_value"] = np.std(img)
        d["filepath"] = study["filepath"]

        return d


# ------------------------------------------------------------------------------------------------ #
#                                    CBIS IMPUTER                                                  #
# ------------------------------------------------------------------------------------------------ #
class CBISImputer:
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

    def fit(self, df: pd.DataFrame) -> CBISImputer:
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
#                                   CBIS TRANSFORMER                                               #
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
    "laterality": {"prefix": "LR", "values": ["LEFT", "RIGHT"]},
    "image_view": {"prefix": "IV", "values": ["CC", "MLO"]},
    "calc_type": {"prefix": "CT", "values": CALC_TYPES},
    "calc_distribution": {"prefix": "CD", "values": CALC_DISTRIBUTIONS},
    "mass_shape": {"prefix": "MS", "values": MASS_SHAPES},
    "mass_margins": {"prefix": "MM", "values": MASS_MARGINS},
}


# ------------------------------------------------------------------------------------------------ #
class CBISTransformer:
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
