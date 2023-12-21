#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/analyze/quality/dicom.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Wednesday December 20th 2023 05:14:46 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Data Quality Module"""
import logging
import sys

import numpy as np
import pandas as pd

from bcd.explore.quality.base import DQA, Validator

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FP_CASE_SERIES_XREF = "data/meta/1_staged/case_series_xref.csv"


# ------------------------------------------------------------------------------------------------ #
class DicomDQA(DQA):
    def __init__(
        self, filepath: str, validator: Validator = Validator, name: str = None
    ) -> None:
        super().__init__(filepath=filepath, name=name)
        self._validator = validator()
        self._df = pd.read_csv(self._filepath)
        self._case_series_xref = pd.read_csv(FP_CASE_SERIES_XREF)
        self._validation_mask = None

    def validate(self) -> np.ndarray:
        "Validates the data and returns a boolean mask of cell validity."
        if self._validation_mask is None:
            suid = self._validator.validate_series_uid(
                series_uid=self._df["series_uid"]
            )
            filepath = self._validator.validate_filepath(filepath=self._df["filepath"])
            pi = self._validator.validate_photometric_interpretation(
                photometric_interpretation=self._df["photometric_interpretation"]
            )
            spp = self._validator.validate_samples_per_pixel(
                samples_per_pixel=self._df["samples_per_pixel"]
            )
            ar = self._validator.validate_aspect_ratio(
                aspect_ratio=self._df["aspect_ratio"]
            )
            height = self._validator.validate_between(
                data=self._df["height"], left=0, right=10000
            )
            width = self._validator.validate_between(
                data=self._df["width"], left=0, right=10000
            )
            bit_depth = self._validator.validate_bit_depth(
                bit_depth=self._df["bit_depth"]
            )
            sip = self._validator.validate_between(
                data=self._df["min_pixel_value"], left=0, right=65535
            )
            lip = self._validator.validate_between(
                data=self._df["max_pixel_value"], left=0, right=65535
            )
            ipr = self._validator.validate_between(
                data=self._df["range_pixel_values"], left=0, right=65535
            )
            size = self._validator.validate_size(df=self._df)
            self._validation_mask = pd.concat(
                [
                    suid,
                    filepath,
                    pi,
                    spp,
                    height,
                    width,
                    size,
                    ar,
                    bit_depth,
                    sip,
                    lip,
                    ipr,
                ],
                axis=1,
            )
            self._validation_mask.columns = [
                "series_uid",
                "filepath",
                "photometric_interpretation",
                "samples_per_pixel",
                "height",
                "width",
                "size",
                "aspect_ratio",
                "bit_depth",
                "min_pixel_value",
                "max_pixel_value",
                "range_pixel_values",
            ]

        return self._validation_mask
