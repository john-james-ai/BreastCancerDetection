#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/dicom/prep.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Friday September 22nd 2023 03:29:16 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Data Prep Module"""
import sys
import os
import logging
from typing import Union
from glob import glob

import pandas as pd
import pydicom

from bcd.data.base import DataPrep

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class DicomPrep(DataPrep):
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
        if result:
            return pd.read_csv(dicom_fp)

    def _get_filepaths(self, location: str, skip_list: list = []) -> list:
        """Returns a filtered list of DICOM filepaths"""
        pattern = location + "/**/*.dcm"
        filepaths = glob(pattern, recursive=True)

        msg = f"Number of filepaths: {len(filepaths)}"
        logger.debug(msg)
        if len(skip_list) > 0:
            filepaths = self._filter_filepaths(filepaths=filepaths, skip_list=skip_list)
        msg = f"Number of filtered filepaths: {len(filepaths)}"
        logger.debug(msg)
        return filepaths

    def _filter_filepaths(self, filepaths: list, skip_list: list) -> bool:
        """Indicates whether a filepath is in the list of files to be skipped"""
        filtered_filepaths = []
        for filepath in filepaths:
            for skipfile in skip_list:
                if not skipfile in filepath:  # noqa
                    filtered_filepaths.append(filepath)
        return filtered_filepaths

    def _extract_dicom_data(self, filepaths: list) -> pd.DataFrame:
        """Extracts dicom data and returns a list of dictionaries."""
        dicom_data = []
        for filepath in filepaths:
            dcm = pydicom.dcmread(filepath)
            dcm_data = {}

            dcm_data["series_uid"] = dcm.SeriesInstanceUID
            dcm_data["filepath"] = os.path.relpath(filepath)
            dcm_data["patient_id"] = self._extract_patient_id(str(dcm.PatientID))
            dcm_data["side"] = self._extract_side(str(dcm.PatientID))
            dcm_data["image_view"] = self._extract_view(str(dcm.PatientID))
            dcm_data["height"] = int(dcm.Rows)
            dcm_data["width"] = int(dcm.Columns)
            dcm_data["bits"] = int(dcm.BitsStored)
            dcm_data["smallest_image_pixel"] = int(dcm.SmallestImagePixelValue)
            dcm_data["largest_image_pixel"] = int(dcm.LargestImagePixelValue)
            dcm_data["image_pixel_range"] = int(dcm.LargestImagePixelValue) - int(
                dcm.SmallestImagePixelValue
            )
            dicom_data.append(dcm_data)
        msg = f"Length of DICOM data: {len(dicom_data)}"
        logger.debug(msg)

        dicom_data = pd.DataFrame(data=dicom_data)
        msg = f"Shape of DICOM Data: {dicom_data.shape}"
        logger.debug(msg)

        return dicom_data

    def _extract_patient_id(self, s: str) -> str:
        """Extracts patient_id from a string

        Args:
            s (str): String containing a patient_id in the form 'P_00000'
        """
        d = "P_"
        return d + s[s.index(d) + len(d) :].split("_")[0]  # noqa

    def _extract_side(self, s: str) -> str:
        """Extracts left or right side from a string

        Args:
            s (str): String containing a patient_id in the form 'P_00000_<side>_<view>'
        """
        d = "P_"
        return s[s.index(d) + len(d) :].split("_")[1]  # noqa

    def _extract_view(self, s: str) -> str:
        """Extracts image view from a string

        Args:
            s (str): String containing a patient_id in the form 'P_00000_<side>_<view>'
        """
        view = "MLO" if "MLO" in s else "CC"
        return view
