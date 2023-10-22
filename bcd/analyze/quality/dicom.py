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
# Modified   : Saturday October 21st 2023 04:16:46 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Data Quality Module"""
import sys
import logging

import pandas as pd
import numpy as np

from bcd.analyze.quality.base import DQA, Validator, DQAResult, Consistency

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FP_CASE_SERIES_XREF = "data/meta/1_staged/case_series_xref.csv"


# ------------------------------------------------------------------------------------------------ #
class DicomDQA(DQA):
    def __init__(self, filepath: str, validator: Validator = Validator, name: str = None) -> None:
        super().__init__(filepath=filepath, name=name)
        self._validator = validator()
        self._df = pd.read_csv(self._filepath)
        self._case_series_xref = pd.read_csv(FP_CASE_SERIES_XREF)
        self._validation_mask = None

    def validate(self) -> np.ndarray:
        "Validates the data and returns a boolean mask of cell validity."
        if self._validation_mask is None:
            suid = self._validator.validate_series_uid(series_uid=self._df["series_uid"])
            filepath = self._validator.validate_filepath(filepath=self._df["filepath"])
            pid = self._validator.validate_patient_id(patient_id=self._df["patient_id"])
            side = self._validator.validate_side(side=self._df["side"])
            image_view = self._validator.validate_image_view(image_view=self._df["image_view"])
            pi = self._validator.validate_photometric_interpretation(
                photometric_interpretation=self._df["photometric_interpretation"]
            )
            spp = self._validator.validate_samples_per_pixel(
                samples_per_pixel=self._df["samples_per_pixel"]
            )
            ar = self._validator.validate_aspect_ratio(aspect_ratio=self._df["aspect_ratio"])
            height = self._validator.validate_between(data=self._df["height"], left=0, right=10000)
            width = self._validator.validate_between(data=self._df["width"], left=0, right=10000)
            bit_depth = self._validator.validate_bit_depth(bit_depth=self._df["bit_depth"])
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
                    pid,
                    side,
                    image_view,
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
                "patient_id",
                "side",
                "image_view",
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

    def analyze_consistency(self) -> DQAResult:
        """Parse the case_id and confirm values match"""

        # Join the with the case/series xref file.
        df = self._df.merge(self._case_series_xref, on="series_uid", how="left")

        pid = df["case_id"].str[:7]
        cid_split = df["case_id"].str.split("_", expand=True)
        side = cid_split[2]
        image_view = cid_split[4]

        pid = df["patient_id"] == pid
        side = df["side"] == side
        image_view = df["image_view"] == image_view

        mask = pd.concat([pid, side, image_view], axis=1)

        # Summary Consistency
        nrows = df.shape[0]
        nrows_consistent = mask.all(axis=1).sum(axis=0)
        row_consistency = round(nrows_consistent / nrows, 3)

        ncells = mask.shape[0] * mask.shape[1]
        ncells_consistent = mask.sum().sum()
        cell_consistency = round(ncells_consistent / ncells, 3)

        sc = Consistency(
            dataset=self._name,
            filename=self._filename,
            records=nrows,
            consistent_records=nrows_consistent,
            record_consistency=row_consistency,
            data_values=ncells,
            consistent_data_values=ncells_consistent,
            data_value_consistency=cell_consistency,
        )

        result = DQAResult(summary=sc, detail=None)
        return result
