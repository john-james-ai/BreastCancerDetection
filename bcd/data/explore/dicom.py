#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/explore/dicom.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:24:00 am                                              #
# Modified   : Tuesday October 17th 2023 12:39:35 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Dataset Module"""
import sys
import os
import logging
from typing import Callable

import pydicom
import matplotlib.pyplot as plt
import pandas as pd

from bcd.data.explore import Dataset

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------------------------ #
DICOM_DTYPES = {
    "series_uid": "str",
    "filepath": "str",
    "patient_id": "str",
    "side": "category",
    "image_view": "category",
    "photometric_interpretation": "category",
    "samples_per_pixel": "int32",
    "height": "int64",
    "width": "int64",
    "size": "int64",
    "aspect_ratio": "float",
    "bits": "category",
    "smallest_image_pixel": "int64",
    "largest_image_pixel": "int64",
    "image_pixel_range": "int64",
    "case_id": "str",
    "series_description": "str",
}

MASS_FEATURES = [
    "cancer",
    "abnormality_type",
    "side",
    "image_view",
    "brisque",
    "breast_density",
    "subtlety",
    "mass_shape",
    "mass_margins",
]
CALC_FEATURES = [
    "cancer",
    "abnormality_type",
    "side",
    "image_view",
    "brisque",
    "breast_density",
    "subtlety",
    "calc_type",
    "calc_distribution",
]


# ------------------------------------------------------------------------------------------------ #
class DicomDataset(Dataset):
    """Dataset containing dicom image metadata

    Args:
        filepath (str): File path to the dataset
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        df = pd.read_csv(self._filepath, dtype=DICOM_DTYPES)
        super().__init__(df=df)

    def plot_images(
        self,
        condition: Callable = None,
        n: int = 4,
        nrows: int = 2,
        ncols: int = 2,
        rowheight: int = 4,
        colwidth: int = 3,
        random_state: int = None,
    ) -> None:
        """Plots a sample of images the meet the designated condition.

        Args:
            condition (Callable): Lambda expression that will be used to subset the DICOM dataset.
            n (int): The number of images to plot
            rows (int): Number of rows to plot
            cols (int): Number of columns to plot
            rowheight (int): Row height in inches.
            colwidth (int): Column width in inches.
            random_state (int): Pseudo random seed.
        """
        n = nrows * ncols
        width = ncols * colwidth
        height = nrows * rowheight
        fig = plt.figure(figsize=(width, height))
        # Filter data as required
        if condition is not None:
            df = super().subset(condition=condition)
        else:
            df = self._df
        df = df.loc[df["series_description"] == "full mammogram images"]
        df.drop_duplicates(inplace=True)
        dfs = round(df.sample(n=n, random_state=random_state), 2)

        for idx, (_, row) in enumerate(dfs.iterrows()):
            filepath = os.path.abspath(row["filepath"])
            fig.add_subplot(nrows, ncols, idx + 1)
            img = pydicom.dcmread(filepath)
            title = self._format_title(row)
            plt.imshow(img.pixel_array, cmap="jet")
            plt.title(title)
        plt.tight_layout()
        plt.show()

    def summary(self) -> pd.DataFrame:
        """Provides a summary of the DICOM Dataset"""
        df = self._df[
            [
                "series_description",
                "height",
                "width",
                "bits",
                "smallest_image_pixel",
                "largest_image_pixel",
                "image_pixel_range",
                "brisque",
            ]
        ]
        return df.groupby(by=["series_description"]).describe()

    def _format_title(self, row: pd.Series) -> str:
        """Formats the title for an image"""
        if row["abnormality_type"] == "calcification":
            features = CALC_FEATURES
        else:
            features = MASS_FEATURES

        title = ""
        for idx, feature in enumerate(features):
            title += f"{feature}: {row[feature]} "
            if idx % 1 == 0:
                title += "\n"
        return title
