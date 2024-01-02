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
# Modified   : Tuesday January 2nd 2024 05:34:21 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Data Prep Module"""
import os

import dask
import numpy as np
import pandas as pd
import pydicom

from bcd.dal.file import IOService
from bcd.data_prep.base import DataPrep
from bcd.utils.file import getsize
from bcd.utils.profile import profiler


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-renamed,arguments-differ
# ------------------------------------------------------------------------------------------------ #
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
