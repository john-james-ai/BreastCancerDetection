#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data_prep/transform.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Monday January 8th 2024 06:07:02 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# pylint: disable=unused-import, arguments-renamed,arguments-differ
# ------------------------------------------------------------------------------------------------ #
"""Data Transform Module"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-renamed,arguments-differ
# ------------------------------------------------------------------------------------------------ #

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
    """One-hot encodes categorical variables.

    The CBIS-DDSM has 45 calcification types, 9 calcification distributions, 20 mass shapes, and
    19 mass margins, many of which are compound categories, in that two or more categories are
    combined. For instance, calcification type 'ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS' indicates
    three different types: 'ROUND_AND_REGULAR', 'PUNCTATE', and 'AMORPHOUS'. Segregating these
    compound categories into separate categories will drastically reduce the number of categories
    to analyze. More importantly, it aligns our data and the analyses with the common morphological
    taxonomy. So, task one is to extract the unary morphological categories from the
    compound classifications.

    Args:
        source_fp (str): Filepath for input dataset
        destination_fp (str): Filepath to output dataset
        force (bool): Whether to force execution if the destination file already exists.
            Default = False.
    """

    def __init__(
        self, source_fp: str, destination_fp: str, force: bool = False
    ) -> None:
        self._source_fp = os.path.abspath(source_fp)
        self._destination_fp = os.path.abspath(destination_fp)
        self._force = force

    def transform(self) -> pd.DataFrame:
        """Performs the transformation of the data."""
        if not os.path.exists(self._destination_fp) or self._force:
            df = pd.read_csv(self._source_fp)
            df["cancer"] = np.where(df["cancer"] == True, 1, 0)  # noqa

            # One-hot encode variables
            df_enc = self._encode_dataset(df=df)
            self._save(df=df_enc)
            return df_enc
        else:
            return pd.read_csv(self._destination_fp)

    def _encode_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encodes the dataset"""
        for feature, data in ENC_VARS.items():
            for value in data["values"]:
                df = self._encode_column(
                    df=df, prefix=data["prefix"], col=feature, value=value
                )
        return df

    def _encode_column(self, df, prefix, col, value):
        "One-hot encodes column"
        newcol = prefix + "_" + value
        df[newcol] = np.where(df[col].str.contains(value), 1, 0)
        return df

    def _save(self, df: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(self._destination_fp), exist_ok=True)
        df.to_csv(self._destination_fp, index=False)
