#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/dataset/case.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:24:00 am                                              #
# Modified   : Saturday September 30th 2023 04:07:55 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Dataset Module"""
import sys
import os
import logging

import pandas as pd

from bcd.data.dataset import Dataset

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------------------------ #
CASE_DTYPES = {
    "case_id": "str",
    "patient_id": "str",
    "breast_density": "int32",
    "left_or_right_breast": "category",
    "image_view": "category",
    "abnormality_id": "int32",
    "abnormality_type": "category",
    "calc_type": "category",
    "calc_distribution": "category",
    "mass_shape": "category",
    "mass_margins": "category",
    "assessment": "int32",
    "pathology": "category",
    "subtlety": "int32",
    "dataset": "category",
    "cancer": "bool",
}
ORDINAL_DTYPES = {
    "breast_density": "category",
    "assessment": "category",
    "subtlety": "category",
}

FEATURES = {
    "breast_density": "ordinal",
    "left_or_right_breast": "nominal",
    "image_view": "nominal",
    "abnormality_type": "nominal",
    "calc_type": "nominal",
    "calc_distribution": "nominal",
    "mass_shape": "nominal",
    "mass_margins": "nominal",
    "assessment": "ordinal",
    "subtlety": "ordinal",
}

CORE_VARIABLES = {
    "breast_density": "ordinal",
    "left_or_right_breast": "nominal",
    "image_view": "nominal",
    "abnormality_type": "nominal",
    "calc_type": "nominal",
    "calc_distribution": "nominal",
    "mass_shape": "nominal",
    "mass_margins": "nominal",
    "assessment": "ordinal",
    "subtlety": "ordinal",
    "cancer": "nominal",
}


# ------------------------------------------------------------------------------------------------ #
class CaseDataset(Dataset):
    """Dataset containing mass cases

    Args:
        filepath (str): File path to the dataset
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        df = pd.read_csv(self._filepath, dtype=CASE_DTYPES)
        super().__init__(df=df)

    def plot_feature_associations(self, *args, **kwargs) -> None:
        """Plots an association matrix showing strength (not direction) of the association between features."""
        df = self._get_feature_association_matrix()
        title = f"CBIS-DDSM Feature Association Plot\nCramer's V"  # noqa
        self.plot.heatmap(data=df, title=title, *args, **kwargs)

    def plot_target_associations(self, *args, **kwargs) -> None:
        df = self._get_target_association_matrix()
        title = f"CBIS-DDSM Target Association Plot\nCramer's V"  # noqa
        self.plot.barplot(data=df, x="strength", y="variable", title=title, *args, **kwargs)

    def get_most_malignant_calc(self, x: str, n: int = 10) -> pd.DataFrame:
        """Returns values of x ordered by proportion of malignant cases

        Args:
            x (str): A categorical independent variable
            n (int): The number of observations to return.

        """
        calc = self._df.loc[self._df["abnormality_type"] == "calcification"]
        return self._get_most_malignant(data=calc, x=x, n=n)

    def get_most_malignant_mass(self, x: str, n: int = 10) -> pd.DataFrame:
        """Returns values of x ordered by proportion of malignant cases

        Args:
            x (str): A categorical independent variable
            n (int): The number of observations to return.

        """
        mass = self._df.loc[self._df["abnormality_type"] == "mass"]
        return self._get_most_malignant(data=mass, x=x, n=n)

    def _get_most_malignant(self, data: pd.DataFrame, x: str, n: int = 10) -> pd.DataFrame:
        prop = (
            data[[x, "cancer"]]
            .groupby(by=[x])
            .value_counts(normalize=True)
            .to_frame()
            .reset_index()
            .sort_values(by="proportion", ascending=False)
        )
        prop = prop.loc[prop["cancer"] == True].nlargest(n, "proportion")  # noqa
        prop[x] = prop[x].astype("object")
        return prop

    def summary(self) -> pd.DataFrame:  # noqa
        """Summarizes the case dataset"""
        d = {}
        d["Patients"] = self._df["patient_id"].nunique()
        d["Cases"] = self._df["case_id"].nunique()
        d["Calcification Cases"] = self._df.loc[
            self._df["abnormality_type"] == "calcification"
        ].shape[0]
        d["Calcification Cases - Benign"] = self._df.loc[
            (self._df["abnormality_type"] == "calcification")
            & (self._df["cancer"] == False)  # noqa
        ].shape[0]
        d["Calcification Cases - Malignant"] = self._df.loc[
            (self._df["abnormality_type"] == "calcification") & (self._df["cancer"] == True)  # noqa
        ].shape[0]

        d["Mass Cases"] = self._df.loc[self._df["abnormality_type"] == "mass"].shape[0]
        d["Mass Cases - Benign"] = self._df.loc[
            (self._df["abnormality_type"] == "mass") & (self._df["cancer"] == False)  # noqa
        ].shape[0]
        d["Mass Cases - Malignant"] = self._df.loc[
            (self._df["abnormality_type"] == "mass") & (self._df["cancer"] == True)  # noqa
        ].shape[0]
        df = pd.DataFrame(data=d, index=[0]).T
        df.columns = ["Summary"]
        return df

    def as_df(self, categorize_ordinals: bool = False) -> pd.DataFrame:
        """Returns the data as a DataFrame, converting the ordinals to category as requested

        Args:
            categorize_ordinals (bool): If True, ordinals are converted to category variables.
                This is done to ensure that the variable is plotted as a categorical
                variable and not numeric.
        """
        if categorize_ordinals:
            return self._df.astype(ORDINAL_DTYPES)
        else:
            return self._df

    def _get_feature_association_matrix(self) -> pd.DataFrame:
        """Creates an association matrix using Cramer's V"""
        matrix = []
        for a in FEATURES.keys():
            vector = []
            for b in FEATURES.keys():
                if a == b:
                    vector.append(1)
                else:
                    result = self.stats.cramersv(a=a, b=b)
                    vector.append(result.value)
            matrix.append(vector)
        matrix = pd.DataFrame(data=matrix, columns=FEATURES.keys(), index=FEATURES.keys())
        return matrix

    def _get_target_association_matrix(self) -> pd.DataFrame:
        """Creates a dependent variable and target association dataframe.."""
        scores = []
        for a in FEATURES.keys():
            result = self.stats.cramersv(a=a, b="cancer")
            d = {"variable": a, "strength": result.value}
            scores.append(d)
        scores = pd.DataFrame(data=scores)
        scores = scores.sort_values(by="strength", ascending=False)
        return scores
