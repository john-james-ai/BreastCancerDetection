#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data_prep/prep.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:38 am                                              #
# Modified   : Wednesday January 3rd 2024 02:24:51 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Prep Module"""
from __future__ import annotations

import os
from glob import glob

import dask
import numpy as np
import pandas as pd
import pydicom

from bcd.dal.file import IOService
from bcd.data_prep.base import DataPrep
from bcd.utils.file import getsize
from bcd.utils.profile import profiler


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-differ
# ------------------------------------------------------------------------------------------------ #
class CasePrep(DataPrep):
    """Performs Case metadata preparation.

    Combines training and test cases into a single csv case file.

    Args:
        calc_train_fp, calc_test_fp, mass_train_fp, mass_test_fp (str): The file paths to the
            calcification and mass training and test sets.
        case_fp (str): Path to output calcification and mass datasets.
        force (bool): Whether to force execution if output already exists. Default is False.
    """

    def __init__(
        self,
        calc_train_fp: str,
        calc_test_fp: str,
        mass_train_fp: str,
        mass_test_fp: str,
        case_fp: str,
        force: bool = False,
    ) -> None:
        super().__init__()
        self._calc_train_fp = calc_train_fp
        self._calc_test_fp = calc_test_fp
        self._mass_train_fp = mass_train_fp
        self._mass_test_fp = mass_test_fp
        self._case_fp = case_fp
        self._force = force

    def prep(self) -> pd.DataFrame:
        """Combines training and test cases into a single csv case file."""

        if self._force or not os.path.exists(self._case_fp):
            # Merge all case data into a single DataFrame
            df_cases = self._merge_cases()

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

            # Create the Boolean target corresponding to pathology
            df_cases["cancer"] = np.where(
                df_cases["pathology"] == "MALIGNANT", True, False
            )

            # Drop the incorrect file path columns.
            columns_to_drop = [
                "image_file_path",
                "cropped_image_file_path",
                "ROI_mask_file_path",
            ]
            df_cases = df_cases.drop(columns=columns_to_drop)

            # Change laterality to laterality, the DICOM attribute
            df_cases = df_cases.rename(columns={"laterality": "laterality"})

            self._save(df=df_cases, filepath=self._case_fp)

            return df_cases

        return pd.read_csv(self._case_fp)

    def _merge_cases(self) -> pd.DataFrame:
        """Combines mass and calcification train and test files into a single file."""
        # Extracts absolute paths, a pre-emptive measure in case
        # jupyter book can't access the path
        calc_train_fp = os.path.abspath(self._calc_train_fp)
        calc_test_fp = os.path.abspath(self._calc_test_fp)
        mass_train_fp = os.path.abspath(self._mass_train_fp)
        mass_test_fp = os.path.abspath(self._mass_test_fp)

        # Read the data
        df_calc_train = pd.read_csv(calc_train_fp)
        df_calc_test = pd.read_csv(calc_test_fp)
        df_mass_train = pd.read_csv(mass_train_fp)
        df_mass_test = pd.read_csv(mass_test_fp)

        # Add the filesets
        df_calc_train["fileset"] = "training"
        df_calc_test["fileset"] = "test"
        df_mass_train["fileset"] = "training"
        df_mass_test["fileset"] = "test"

        # Standardize column names with underscores in place of spaces.
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
            + df["laterality"].apply(lambda x: x.upper())
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
#                                   SERIES PREP                                                    #
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


# ------------------------------------------------------------------------------------------------ #
#                                     CBIS PREP                                                    #
# ------------------------------------------------------------------------------------------------ #
class CBISPrep(DataPrep):
    """Extracts DICOM data and integrates it with a single CBIS dataset.

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
