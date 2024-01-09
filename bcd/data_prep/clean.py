#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data_prep/clean.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Monday January 8th 2024 04:27:35 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# pylint: disable=unused-import,arguments-renamed,arguments-differ,cell-var-from-loop
# ------------------------------------------------------------------------------------------------ #
"""Data Cleaning Module"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from studioai.preprocessing.encode import RankFrequencyEncoder


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
            missing_values=np.nan,
            add_indicator=True,
            n_nearest_features=None,
            sample_posterior=True,
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
        for col in df.columns:
            values = np.array(sorted(self._encoded_values[col]))
            df[col] = df[col].apply(lambda x: values[np.argmin(np.abs(x - values))])
        return df
