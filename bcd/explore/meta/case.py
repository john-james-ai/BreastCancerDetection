#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/explore/meta/case.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:24:00 am                                              #
# Modified   : Tuesday January 2nd 2024 04:19:19 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Dataset Module"""
import logging
import os
import sys
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from studioai.analysis.visualize.visualizer import SeabornCanvas

from bcd.explore.meta import Explorer
from bcd.utils.string import proper

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sns.set_style("whitegrid")
# ------------------------------------------------------------------------------------------------ #
CALCIFICATION_DATA = [
    "mmg_id",
    "patient_id",
    "breast_density",
    "laterality",
    "image_view",
    "abnormality_id",
    "calc_type",
    "calc_distribution",
    "assessment",
    "pathology",
    "subtlety",
    "fileset",
    "cancer",
]
MASS_DATA = [
    "mmg_id",
    "patient_id",
    "breast_density",
    "laterality",
    "image_view",
    "abnormality_id",
    "mass_shape",
    "mass_margins",
    "assessment",
    "pathology",
    "subtlety",
    "fileset",
    "cancer",
]
CASE_DTYPES = {
    "mmg_id": "str",
    "patient_id": "str",
    "breast_density": "int32",
    "laterality": "category",
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
    "fileset": "category",
    "cancer": "bool",
}
ORDINAL_DTYPES = {
    "breast_density": "category",
    "assessment": "category",
    "subtlety": "category",
}
FEATURES = {
    "breast_density": "ordinal",
    "laterality": "nominal",
    "image_view": "nominal",
    "abnormality_type": "nominal",
    "calc_type": "nominal",
    "calc_distribution": "nominal",
    "mass_shape": "nominal",
    "mass_margins": "nominal",
    "assessment": "ordinal",
    "subtlety": "ordinal",
}
MODEL_FEATURES = [
    "AT_calcification",
    "AT_mass",
    "breast_density",
    "CD_CLUSTERED",
    "CD_DIFFUSELY_SCATTERED",
    "CD_LINEAR",
    "CD_REGIONAL",
    "CD_SEGMENTAL",
    "CT_AMORPHOUS",
    "CT_COARSE",
    "CT_DYSTROPHIC",
    "CT_EGGSHELL",
    "CT_FINE_LINEAR_BRANCHING",
    "CT_LARGE_RODLIKE",
    "CT_LUCENT_CENTERED",
    "CT_MILK_OF_CALCIUM",
    "CT_PLEOMORPHIC",
    "CT_PUNCTATE",
    "CT_ROUND_AND_REGULAR",
    "CT_SKIN",
    "CT_VASCULAR",
    "IV_CC",
    "IV_MLO",
    "LR_LEFT",
    "LR_RIGHT",
    "MM_CIRCUMSCRIBED",
    "MM_ILL_DEFINED",
    "MM_MICROLOBULATED",
    "MM_OBSCURED",
    "MM_SPICULATED",
    "MS_ARCHITECTURAL_DISTORTION",
    "MS_ASYMMETRIC_BREAST_TISSUE",
    "MS_FOCAL_ASYMMETRIC_DENSITY",
    "MS_IRREGULAR",
    "MS_LOBULATED",
    "MS_LYMPH_NODE",
    "MS_OVAL",
    "MS_ROUND",
    "subtlety",
]
CALC_MODEL_FEATURES = [
    "breast_density",
    "CD_CLUSTERED",
    "CD_DIFFUSELY_SCATTERED",
    "CD_LINEAR",
    "CD_REGIONAL",
    "CD_SEGMENTAL",
    "CT_AMORPHOUS",
    "CT_COARSE",
    "CT_DYSTROPHIC",
    "CT_EGGSHELL",
    "CT_FINE_LINEAR_BRANCHING",
    "CT_LARGE_RODLIKE",
    "CT_LUCENT_CENTERED",
    "CT_MILK_OF_CALCIUM",
    "CT_PLEOMORPHIC",
    "CT_PUNCTATE",
    "CT_ROUND_AND_REGULAR",
    "CT_SKIN",
    "CT_VASCULAR",
    "IV_CC",
    "IV_MLO",
    "LR_LEFT",
    "LR_RIGHT",
    "subtlety",
]
MASS_MODEL_FEATURES = [
    "breast_density",
    "IV_CC",
    "IV_MLO",
    "LR_LEFT",
    "LR_RIGHT",
    "MM_CIRCUMSCRIBED",
    "MM_ILL_DEFINED",
    "MM_MICROLOBULATED",
    "MM_OBSCURED",
    "MM_SPICULATED",
    "MS_ARCHITECTURAL_DISTORTION",
    "MS_ASYMMETRIC_BREAST_TISSUE",
    "MS_FOCAL_ASYMMETRIC_DENSITY",
    "MS_IRREGULAR",
    "MS_LOBULATED",
    "MS_LYMPH_NODE",
    "MS_OVAL",
    "MS_ROUND",
    "subtlety",
]
CALC_FEATURES = {
    "breast_density": "ordinal",
    "laterality": "nominal",
    "image_view": "nominal",
    "calc_type": "nominal",
    "calc_distribution": "nominal",
    "assessment": "ordinal",
    "subtlety": "ordinal",
}
MASS_FEATURES = {
    "breast_density": "ordinal",
    "laterality": "nominal",
    "image_view": "nominal",
    "mass_shape": "nominal",
    "mass_margins": "nominal",
    "assessment": "ordinal",
    "subtlety": "ordinal",
}

MORPHOLOGY_PREFIX = {
    "calc_type": "CT_",
    "calc_distribution": "CD_",
    "mass_shape": "MS_",
    "mass_margins": "MM_",
}


# ------------------------------------------------------------------------------------------------ #
class CaseExplorer(Explorer):
    """Encapsulates Case Data

    Can be instantiated from a DataFrame or a file path to the data. If both are provided,
    the DataFrame will be used as the data and the file path will be considered
    the persistence location.

    Args:
        df (pd.DataFrame): DataFrame containing cases. Optional. If not provided,
            the filepath  will be used to obtain the data.
        filepath (str): File path to the dataset. Optional. If not provided,
            the df parameter must not be None.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        filepath: str = None,
        canvas: SeabornCanvas = SeabornCanvas(),
    ) -> None:
        if df is None and filepath is None:
            msg = "Must provide 'df' and/or 'filepath' parameters."
            raise ValueError(msg)
        if df is None:
            self._filepath = os.path.abspath(filepath)
            df = pd.read_csv(self._filepath, dtype=CASE_DTYPES)
        super().__init__(df=df)
        self._canvas = canvas

    @property
    def summary(self) -> pd.DataFrame:  # noqa
        """Summarizes the case dataset"""
        d = {}
        d["Patients"] = self._df["patient_id"].nunique()
        d["Cases"] = self._df["mmg_id"].nunique()
        d["Calcification Cases"] = self._df.loc[
            self._df["abnormality_type"] == "calcification"
        ].shape[0]
        d["Calcification Cases - Benign"] = self._df.loc[
            (self._df["abnormality_type"] == "calcification")
            & (self._df["cancer"] == False)  # noqa
        ].shape[0]
        d["Calcification Cases - Malignant"] = self._df.loc[
            (self._df["abnormality_type"] == "calcification")
            & (self._df["cancer"] == True)  # noqa
        ].shape[0]

        d["Mass Cases"] = self._df.loc[self._df["abnormality_type"] == "mass"].shape[0]
        d["Mass Cases - Benign"] = self._df.loc[
            (self._df["abnormality_type"] == "mass")
            & (self._df["cancer"] == False)  # noqa
        ].shape[0]
        d["Mass Cases - Malignant"] = self._df.loc[
            (self._df["abnormality_type"] == "mass")
            & (self._df["cancer"] == True)  # noqa
        ].shape[0]
        df = pd.DataFrame(data=d, index=[0]).T
        df.columns = ["Summary"]
        return df

    def get_calc_data(self) -> pd.DataFrame:
        df = self._df.loc[self._df["abnormality_type"] == "calcification"]
        return df[CALCIFICATION_DATA]

    def get_mass_data(self) -> pd.DataFrame:
        df = self._df.loc[self._df["abnormality_type"] == "mass"]
        return df[MASS_DATA]

    def get_model_data(self) -> tuple:
        """Returns model data for both calcification and mass cases

        Returns:
            Tuple containing train test splits.
        """
        X_train = self._df.loc[self._df["fileset"] == "train"][MODEL_FEATURES]
        y_train = self._df.loc[self._df["fileset"] == "train"]["cancer"]
        X_test = self._df.loc[self._df["fileset"] == "test"][MODEL_FEATURES]
        y_test = self._df.loc[self._df["fileset"] == "test"]["cancer"]
        return (X_train, y_train, X_test, y_test)

    def get_calc_model_data(self) -> tuple:
        """Returns model data for calcification cases

        Returns:
            Tuple containing train test splits.
        """
        X_train = self._df.loc[
            (self._df["abnormality_type"] == "calcification")
            & (self._df["fileset"] == "train")
        ][CALC_MODEL_FEATURES]
        y_train = self._df.loc[
            (self._df["abnormality_type"] == "calcification")
            & (self._df["fileset"] == "train")
        ]["cancer"]
        X_test = self._df.loc[
            (self._df["abnormality_type"] == "calcification")
            & (self._df["fileset"] == "test")
        ][CALC_MODEL_FEATURES]
        y_test = self._df.loc[
            (self._df["abnormality_type"] == "calcification")
            & (self._df["fileset"] == "test")
        ]["cancer"]
        return (X_train, y_train, X_test, y_test)

    def get_mass_model_data(self) -> tuple:
        """Returns model data for mass cases

        Returns:
            Tuple containing train test splits.
        """
        X_train = self._df.loc[
            (self._df["abnormality_type"] == "mass") & (self._df["fileset"] == "train")
        ][MASS_MODEL_FEATURES]
        y_train = self._df.loc[
            (self._df["abnormality_type"] == "mass") & (self._df["fileset"] == "train")
        ]["cancer"]
        X_test = self._df.loc[
            (self._df["abnormality_type"] == "mass") & (self._df["fileset"] == "test")
        ][MASS_MODEL_FEATURES]
        y_test = self._df.loc[
            (self._df["abnormality_type"] == "mass") & (self._df["fileset"] == "test")
        ]["cancer"]
        return (X_train, y_train, X_test, y_test)

    def plot_feature_associations(self, *args, **kwargs) -> None:
        """Plots an association matrix showing strength (not direction) of the association between features."""
        df = self._get_feature_association_matrix(features=FEATURES)
        title = "CBIS-DDSM Case Feature Association Plot\nCramer's V"  # noqa
        return self.plot.heatmap(data=df, title=title, *args, **kwargs)

    def plot_mass_feature_associations(self, *args, **kwargs) -> None:
        """Plots an association matrix showing strength (not direction) of the association between features."""
        df = self._get_feature_association_matrix(features=MASS_FEATURES)
        title = "CBIS-DDSM Mass Case Feature Association Plot\nCramer's V"  # noqa
        return self.plot.heatmap(data=df, title=title, *args, **kwargs)

    def plot_calc_feature_associations(self, *args, **kwargs) -> None:
        """Plots an association matrix showing strength (not direction) of the association between features."""
        df = self._get_feature_association_matrix(features=CALC_FEATURES)
        title = (
            "CBIS-DDSM Calcification Case Feature Association Plot\nCramer's V"  # noqa
        )
        return self.plot.heatmap(data=df, title=title, *args, **kwargs)

    def plot_target_associations(self, *args, **kwargs) -> None:
        df = self._get_target_association_matrix()
        title = "CBIS-DDSM Target Association Plot\nCramer's V"  # noqa
        return self.plot.barplot(
            data=df, x="strength", y="variable", title=title, *args, **kwargs
        )

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

    def summarize_morphology_by_feature(
        self, morphology: str, by: str, figsize: tuple = (12, 8)
    ) -> pd.DataFrame:
        """Summarizes calcification or mass morphology by a feature.

        Morphology data are summarized in terms of the proportion of records
        in the dataset for the values of a feature designated by the 'by' variable.

        Args:
            morphology (str): Calcification or mass morphology variable.
            by (str): A (non-morphology) feature in the dataset.
            figsize (tuple): Tuple containing (maximum row width, height of single row)
        """
        summary = pd.DataFrame()
        prefix = MORPHOLOGY_PREFIX[morphology]
        df = self._df
        df[by] = df[by].astype("category")
        morph_columns = df.columns[df.columns.str.contains(prefix)].values
        for col in morph_columns:
            dfm = df.loc[df[col] == 1]
            dfm_summary = dfm[by].value_counts(normalize=True).to_frame()
            dfm_summary[morphology] = col.replace(prefix, "")
            summary = pd.concat([summary, dfm_summary], axis=0)
        summary = (
            summary.groupby(by=[morphology, by])
            .sum()
            .sort_values(by=[morphology, "proportion"], ascending=[True, False])
        )
        fig = self._plot_morphology_by_feature(
            df=summary, morphology=morphology, by=by, figsize=figsize
        )

        plt.close()

        return fig, summary

    def compare_morphology(
        self, *args, m1: str, m2: str, figsize: tuple = (12, 8), **kwargs
    ) -> Union[plt.Axes, pd.DataFrame]:
        """Compares two morphologies, providing proportions in which m2 is present for m1

        Args:
            m1,m2 (str): Complementary morphologies, e.g., mass_shape and mass_margins
            figsize (tuple): Tuple containing (max width, row_height)
        """
        df = self._df
        proportions = []
        m1_cols = [col for col in df.columns if col.__contains__(MORPHOLOGY_PREFIX[m1])]
        m2_cols = [col for col in df.columns if col.__contains__(MORPHOLOGY_PREFIX[m2])]
        for a in m1_cols:
            arows = len(df.loc[df[a] == 1])
            for b in m2_cols:
                abrows = len(df.loc[(df[a] == 1) & (df[b] == 1)])
                d = {
                    m1: a.replace(MORPHOLOGY_PREFIX[m1], ""),
                    m2: b.replace(MORPHOLOGY_PREFIX[m2], ""),
                    "proportion": abrows / arows,
                }
                proportions.append(d)
        comparison = pd.DataFrame(data=proportions).groupby(by=[m1, m2]).sum()

        # Drop rows with proportion = 0
        comparison = comparison.loc[comparison["proportion"] > 0]

        fig = self._plot_morphology_by_feature(
            df=comparison, morphology=m1, by=m2, figsize=figsize, *args, **kwargs
        )

        plt.close()

        return fig, comparison

    def morphology_analysis(self, a: str, b: str, figsize: tuple = (8, 4)) -> plt.Axes:
        """Returns a heatmap of malignancy probabilities for pairs of morphological features

        Args:
            a,b (str): Morphological features
            figsize (tuple): Width and height of figure in inches.
        """
        # Filter morphologies to include those most frequently encountered
        if "mass" in a:
            df = self._get_mass_morphology()
        else:
            df = self._get_calc_morphology()

        # Compute the number of malignancies by group and normalize
        # by the total number of malignancies for the base feature.
        g = df.groupby(by=[a, b])["cancer"].sum()
        g = g / df["cancer"].sum()
        g = g.to_frame().reset_index()

        # Pivot the dataframe such that the base feature
        # is the index and the columns are values of
        # the additive feature.
        g = g.pivot(index=a, columns=b, values="cancer").fillna(0)

        _, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(g, cmap="crest", annot=True)

        title = f"Probability of Malignancy by {proper(a)} and {proper(b)}."
        _ = ax.set_title(title)

        return ax

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

    def _get_feature_association_matrix(self, features: dict) -> pd.DataFrame:
        """Creates an association matrix using Cramer's V"""
        matrix = []
        for a in features.keys():
            vector = []
            for b in features.keys():
                if a == b:
                    vector.append(1)
                else:
                    result = self.stats.cramersv(a=a, b=b)
                    vector.append(result.value)
            matrix.append(vector)
        matrix = pd.DataFrame(
            data=matrix, columns=features.keys(), index=features.keys()
        )
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

    def _get_most_malignant(
        self, data: pd.DataFrame, x: str, n: int = 10
    ) -> pd.DataFrame:
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

    def _plot_morphology_by_feature(
        self,
        df: pd.DataFrame,
        morphology: str,
        by: str,
        figsize: tuple,
        *args,
        **kwargs,
    ) -> None:
        """Plots morphology proportional counts by feature."""
        df2 = df.reset_index()

        nplots = df2[morphology].nunique()
        fig, axes = self._canvas.get_figaxes(nplots=nplots, figsize=figsize)

        title = proper(f"{morphology} and {by}")
        suptitle = f"CBIS-DDSM Morphology Analysis\n{title}"

        for ax, (value, group) in zip(axes, df2.groupby(by=morphology)):
            self.plot.barplot(
                data=group, x=by, y="proportion", ax=ax, title=value, *args, **kwargs
            )

        _ = fig.suptitle(suptitle)
        plt.close()

        return fig

    def _get_mass_morphology(self) -> pd.DataFrame:
        """Returns a DataFrame containing the top mass morphologies"""
        # Filter to include theh top five mass shapes and margins
        mass_shapes = [
            "IRREGULAR",
            "OVAL",
            "LOBULATED",
            "ROUND",
            "ARCHITECTURAL_DISTORTION",
        ]
        mass_margins = [
            "SPICULATED",
            "CIRCUMSCRIBED",
            "ILL_DEFINED",
            "OBSCURED",
            "MICROLOBULATED",
        ]
        df = self._df.loc[self._df["mass_shape"].isin(mass_shapes)]
        df = df.loc[df["mass_margins"].isin(mass_margins)]
        df["mass_shape"] = df["mass_shape"].astype(str)
        df["mass_margins"] = df["mass_margins"].astype(str)
        return df

    def _get_calc_morphology(self) -> pd.DataFrame:
        """Returns a DataFrame containing the top calc morphologies"""
        calc_types = [
            "PLEOMORPHIC",
            "AMORPHOUS",
            "PUNCTATE",
            "LUCENT_CENTERED",
            "FINE_LINEAR_BRANCHING",
            "VASCULAR",
        ]
        calc_distributions = [
            "CLUSTERED",
            "SEGMENTAL",
            "LINEAR",
            "REGIONAL",
            "DIFFUSELY_SCATTERED",
        ]
        df = self._df.loc[self._df["calc_type"].isin(calc_types)]
        df = df.loc[df["calc_distribution"].isin(calc_distributions)]
        df["calc_type"] = df["calc_type"].astype(str)
        df["calc_distribution"] = df["calc_distribution"].astype(str)
        return df
