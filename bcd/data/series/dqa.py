#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/series/dqa.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:51 am                                              #
# Modified   : Saturday September 23rd 2023 03:36:49 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Series Data Quality Module"""
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
class SeriesDQA(DQA):
    def __init__(self, filepath: str, validator: Validator = Validator, name: str = None) -> None:
        super().__init__(filepath=filepath, name=name)
        self._validator = validator()
        self._df = pd.read_csv(self._filepath)
        self._validation_mask = None

    def validate(self) -> np.ndarray:
        "Validates the data and returns a boolean mask of cell validity."
        if self._validation_mask is None:
            suid = self._validator.validate_series_uid(series_uid=self._df["series_uid"])
            desc = self._validator.validate_series_description(
                series_description=self._df["series_description"]
            )
            images = self._validator.validate_between(
                data=self._df["number_of_images"], left=1, right=10
            )
            floc = self._validator.validate_filepath(filepath=self._df["file_location"])
            self._validation_mask = pd.concat(
                [
                    suid,
                    desc,
                    images,
                    floc,
                ],
                axis=1,
            )
            self._validation_mask.columns = [
                "series_uid",
                "series_description",
                "number_of_images",
                "file_location",
            ]

        return self._validation_mask
