#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data_prep/dicom.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Monday January 1st 2024 03:17:32 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Data Prep Module"""
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
# pylint: disable=arguments-renamed,arguments-differ
# ------------------------------------------------------------------------------------------------ #
class DicomPrep(DataPrep):
    """Performs extraction of the DICOM data.

    Iterates through the DICOM metadata in parallel, extracting subject, series, and file location
    data. Then each DICOM file in the directory is parsed and the results are
    combined into to a DataFrame and saved.

    Args:
        filepath (str): Path to the DICOM series metadata.
        dicom_filepath (str) Path for the results
        skip_list (list): List of filepaths to skip.
        force (bool): Whether to force execution if output already exists. Default is False.
    """

    __BASEDIR = "data/image/0_raw/"

    def __init__(
        self,
        filepath: str,
        dicom_filepath: str,
        skip_list: list = None,
        force: bool = False,
    ) -> None:
        super().__init__()
        self._filepath = os.path.abspath(filepath)
        self._dicom_filepath = os.path.abspath(dicom_filepath)
        self._skip_list = skip_list
        self._force = force

    @profiler
    def prep(self) -> pd.DataFrame:
        """Extracts image metadata from the DICOM image files."""

        if self._force or not os.path.exists(self._dicom_filepath):
            # Reads the series metadata that contains subject, series, and
            # file location information
            studies = IOService.read(self._filepath)

            # Add filepaths to the study data first to avoid batch
            # operation exceptions with dask.
            studies = self._get_filepaths(studies=studies)

            results = []
            # Graph of work is created and executed lazily at compute time.
            for study in studies:
                image_result = dask.delayed(self._extract_data)(study)
                results.append(image_result)

            # Compute the results and convert to dataframe
            results = dask.compute(*results)
            df = pd.DataFrame(data=results)

            self._save(df=df, filepath=self._dicom_filepath)

            return df

        return pd.read_csv(self._dicom_filepath)

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

    def _extract_data(self, study: pd.Series) -> dict:
        """Reads study and dicom data from a file."""

        # Parse the study id
        studyid = study["subject_id"].split("_")[0:5]
        abtype, fileset = studyid[0].split("-")

        # Extract the DICOM data
        dcm = pydicom.dcmread(study["filepath"])
        img = dcm.pixel_array

        d = {}
        d["patient_id"] = ("_").join(studyid[1:3])
        d["subject_id"] = study["subject_id"]
        d["abnormality_type"] = abtype.lower()
        d["laterality"] = studyid[3]
        d["image_view"] = studyid[4]
        d["fileset"] = fileset.lower()
        d["series_description"] = study["series_description"]
        d["photometric_interpretation"] = dcm.PhotometricInterpretation
        d["bit_depth"] = dcm.BitsStored
        d["rows"], d["cols"] = img.shape
        d["aspect_ratio"] = d["cols"] / d["rows"]
        d["size"] = d["rows"] * d["cols"]
        d["file_size"] = getsize(study["filepath"])
        d["min_pixel_value"] = dcm.SmallestImagePixelValue
        d["max_pixel_value"] = dcm.LargestImagePixelValue
        d["mean_pixel_value"] = np.mean(img)
        d["std_pixel_value"] = np.std(img)
        d["filepath"] = study["filepath"]
        d["mmg_id"] = "_".join(study["subject_id"].split("_")[0:5])

        return d
