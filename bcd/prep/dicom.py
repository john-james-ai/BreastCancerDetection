#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/prep/dicom.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Thursday October 26th 2023 08:28:24 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Data Prep Module"""
import os
import logging
from typing import Union
from glob import glob
import uuid

import joblib
import numpy as np
from tqdm import tqdm
import pandas as pd
import pydicom

from bcd.prep.base import DataPrep


# ------------------------------------------------------------------------------------------------ #
class DicomPrep(DataPrep):
    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def prep(
        self,
        location: str,
        dicom_fp: str,
        skip_list: list = [],
        force: bool = False,
        result: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """Extracts image metadata from the DICOM image files.

        Args:
            location (str): The base location for the DICOM image files.
            dicom_fp (str) Filename for the dicom metadata dataset.
            skip_list (list): List of filepaths relative to the location to skip.
            force (bool): Whether to force execution if output already exists. Default is False.
            result (bool): Whether the result should be returned. Default is False.
        """
        location = os.path.abspath(location)
        dicom_fp = os.path.abspath(dicom_fp)

        os.makedirs(os.path.dirname(dicom_fp), exist_ok=True)

        if force or not os.path.exists(dicom_fp):
            filepaths = self._get_filepaths(location, skip_list)
            dicom_data = self._extract_dicom_data(filepaths=filepaths)
            dicom_data.to_csv(dicom_fp, index=False)
            msg = f"Shape of DICOM Data: {dicom_data.shape}"
            self._logger.debug(msg)

        if result:
            return pd.read_csv(dicom_fp)

    def merge_case_data(self, case_fp: str, dicom_fp: str, xref_fp: str, dicom_out_fp: str) -> None:
        """Merges case data into the DICOM metadata

        Args:
            case_fp (str): Case data filepath
            dicom_fp (str): DICOM input filepath
            xref_fp (str): Filepath to the case/series x-ref data
            dicom_out_fp (str): Filepath to the DICOM output file.
        """
        CASE_VARIABLES = [
            "case_id",
            "left_or_right_breast",
            "image_view",
            "abnormality_type",
            "assessment",
            "breast_density",
            "calc_type",
            "calc_distribution",
            "mass_shape",
            "mass_margins",
            "fileset",
            "cancer",
        ]
        df_case = pd.read_csv(case_fp, usecols=CASE_VARIABLES)
        df_dicom = pd.read_csv(dicom_fp)
        df_xref = pd.read_csv(xref_fp)

        df_case = df_case.merge(df_xref, on="case_id")
        df_dicom = df_dicom.merge(df_case, on=["series_uid", "series_description"])
        df_dicom.to_csv(dicom_out_fp, index=False)

    def add_series_description(self, dicom_fp, series_fp: str) -> None:
        """Adds series description to the DICOM data

        Args:
            dicom_fp (str) Filename for the dicom metadata dataset.
            series_fp (str): Filepath to the series description data.

        Returns:
            Dataset with series description added.
        """
        dicom = pd.read_csv(dicom_fp)
        if "series_description" not in dicom.columns:
            series = pd.read_csv(series_fp)
            series = series[["series_uid", "series_description"]].drop_duplicates()
            dicom = dicom.merge(series, on="series_uid", how="left")
            dicom.to_csv(dicom_fp, index=False)
        return dicom

    def _get_filepaths(self, location: str, skip_list: list = []) -> list:
        """Returns a filtered list of DICOM filepaths"""
        pattern = location + "/**/*.dcm"
        filepaths = glob(pattern, recursive=True)

        msg = f"Number of filepaths: {len(filepaths)}"
        self._logger.debug(msg)
        if len(skip_list) > 0:
            filepaths = self._filter_filepaths(filepaths=filepaths, skip_list=skip_list)
        msg = f"Number of filtered filepaths: {len(filepaths)}"
        self._logger.debug(msg)
        return filepaths

    def _filter_filepaths(self, filepaths: list, skip_list: list) -> bool:
        """Indicates whether a filepath is in the list of files to be skipped"""
        filtered_filepaths = []
        for filepath in filepaths:
            for skipfile in skip_list:
                if not skipfile in filepath:  # noqa
                    filtered_filepaths.append(filepath)
        return filtered_filepaths

    def _read_dicom_data(self, filepath) -> dict:
        """Reads dicom data from a file."""

        dcm = pydicom.dcmread(filepath)
        img = dcm.pixel_array

        dcm_data = {}
        dcm_data["uid"] = str(uuid.uuid4())
        dcm_data["series_uid"] = dcm.SeriesInstanceUID
        dcm_data["filepath"] = os.path.relpath(filepath)
        dcm_data["photometric_interpretation"] = dcm.PhotometricInterpretation
        dcm_data["samples_per_pixel"] = int(dcm.SamplesPerPixel)
        dcm_data["height"] = int(dcm.Rows)
        dcm_data["width"] = int(dcm.Columns)
        dcm_data["size"] = int(dcm.Columns) * int(dcm.Rows)
        dcm_data["aspect_ratio"] = int(dcm.Columns) / int(dcm.Rows)
        dcm_data["bit_depth"] = int(dcm.BitsStored)
        dcm_data["min_pixel_value"] = int(dcm.SmallestImagePixelValue)
        dcm_data["max_pixel_value"] = int(dcm.LargestImagePixelValue)
        dcm_data["range_pixel_values"] = int(dcm.LargestImagePixelValue) - int(
            dcm.SmallestImagePixelValue
        )
        dcm_data["mean_pixel_value"] = np.mean(img, axis=None)
        dcm_data["median_pixel_value"] = np.median(img, axis=None)
        dcm_data["std_pixel_value"] = np.std(img, axis=None)

        return dcm_data

    def _extract_dicom_data(self, filepaths: list, n_jobs: int = 6) -> pd.DataFrame:
        """Extracts dicom data and returns a DataFrame."""
        dicom_data = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self._read_dicom_data)(filepath) for filepath in tqdm(filepaths)
        )
        dicom_data = pd.DataFrame(data=dicom_data)

        return dicom_data
