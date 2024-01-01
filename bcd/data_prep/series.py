#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data_prep/series.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Monday January 1st 2024 05:44:56 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Data Prep Module"""
import os
from glob import glob

import pandas as pd

from bcd.dal.file import IOService
from bcd.data_prep.base import DataPrep
from bcd.utils.profile import profiler


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-renamed,arguments-differ
# ------------------------------------------------------------------------------------------------ #
class SeriesPrep(DataPrep):
    """Adds filepaths to the Series dataset

    Args:
        filepath (str): Path to the DICOM series metadata.
        series_filepath (str) Path for the results
        force (bool): Whether to force execution if output already exists. Default is False.
    """

    __BASEDIR = "data/image/0_raw/"

    def __init__(
        self,
        filepath: str,
        series_filepath: str,
        force: bool = False,
    ) -> None:
        super().__init__()
        self._filepath = os.path.abspath(filepath)
        self._series_filepath = os.path.abspath(series_filepath)
        self._force = force

    @profiler
    def prep(self) -> pd.DataFrame:
        """Extracts image metadata from the DICOM image files."""

        if self._force or not os.path.exists(self._series_filepath):
            # Reads the series metadata that contains subject, series, and
            # file location information
            studies = IOService.read(self._filepath)

            # Add filepaths to the study data first to avoid batch
            # operation exceptions with dask.
            studies = self._get_filepaths(studies=studies)

            df = pd.DataFrame(data=studies)

            self._save(df=df, filepath=self._series_filepath)

            return df

        return pd.read_csv(self._series_filepath)

    def _get_filepaths(self, studies: pd.Series) -> pd.DataFrame:
        """Adds filepaths to the studies dataframe"""
        studies_filepaths = []
        for _, row in studies.iterrows():
            location = row["file_location"].replace("./", "")
            filepath = os.path.join(self.__BASEDIR, location)
            filepaths = glob(filepath + "/*.dcm")
            for file in filepaths:
                row["filepath"] = file
                studies_filepaths.append(row)
        return studies_filepaths
