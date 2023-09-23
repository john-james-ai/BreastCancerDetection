#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/dicom/dqa.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:25:33 am                                              #
# Modified   : Saturday September 23rd 2023 03:25:15 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""DICOM Data Quality Module"""
import sys
import logging

import pandas as pd
import numpy as np

from bcd.data.dqa import DQA, Validator

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class DicomDQA(DQA):
    def __init__(self, filepath: str, validator: Validator = Validator, name: str = None) -> None:
        super().__init__(filepath=filepath, name=name)
        self._validator = validator()
        self._df = pd.read_csv(self._filepath)
        self._validation_mask = None

    def validate(self) -> np.ndarray:
        "Validates the data and returns a boolean mask of cell validity."
        if self._validation_mask is None:
            suid = self._validator.validate_series_uid(series_uid=self._df["series_uid"])
            filepath = self._validator.validate_filepath(filepath=self._df["filepath"])
            pid = self._validator.validate_patient_id(patient_id=self._df["patient_id"])
            side = self._validator.validate_side(side=self._df["side"])
            view = self._validator.validate_image_view(image_view=self._df["image_view"])
            height = self._validator.validate_between(data=self._df["height"], left=0, right=10000)
            width = self._validator.validate_between(data=self._df["width"], left=0, right=10000)
            bits = self._validator.validate_image_bits(image_bits=self._df["bits"])
            sip = self._validator.validate_between(
                data=self._df["smallest_image_pixel"], left=0, right=65535
            )
            lip = self._validator.validate_between(
                data=self._df["largest_image_pixel"], left=0, right=65535
            )
            ipr = self._validator.validate_between(
                data=self._df["image_pixel_range"], left=0, right=65535
            )
            self._validation_mask = pd.concat(
                [
                    suid,
                    filepath,
                    pid,
                    side,
                    view,
                    height,
                    width,
                    bits,
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
                "height",
                "width",
                "bits",
                "smallest_image_pixel",
                "largest_image_pixel",
                "image_pixel_range",
            ]

        return self._validation_mask
