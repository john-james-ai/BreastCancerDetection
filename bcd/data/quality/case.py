#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/quality/case.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:51 am                                              #
# Modified   : Wednesday October 18th 2023 11:14:51 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Data Quality Module"""
import sys
import logging

import pandas as pd
import numpy as np

from bcd.data.quality.base import DQA, Validator

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class CaseDQA(DQA):
    def __init__(self, filepath: str, validator: Validator = Validator, name: str = None) -> None:
        super().__init__(filepath=filepath, name=name)
        self._validator = validator()
        self._df = pd.read_csv(self._filepath)
        self._validation_mask = None

    def validate(self) -> np.ndarray:
        "Validates the data and returns a boolean mask of cell validity."
        if self._validation_mask is None:
            cid = self._validator.validate_case_id(self._df)
            pid = self._validator.validate_patient_id(self._df["patient_id"])
            bd = self._validator.validate_breast_density(self._df["breast_density"])
            side = self._validator.validate_side(self._df["left_or_right_breast"])
            view = self._validator.validate_image_view(self._df["image_view"])
            aid = self._validator.validate_between(self._df["abnormality_id"], left=1, right=10)
            at = self._validator.validate_abnormality_type(self._df["abnormality_type"])
            ct = self._validator.validate_calc_type(self._df["calc_type"])
            cd = self._validator.validate_calc_distribution(self._df["calc_distribution"])
            ms = self._validator.validate_mass_shape(self._df["mass_shape"])
            mm = self._validator.validate_mass_margins(self._df["mass_margins"])
            asmt = self._validator.validate_assessment(self._df["assessment"])
            path = self._validator.validate_pathology(self._df["pathology"])
            sub = self._validator.validate_subtlety(self._df["subtlety"])
            fs = self._validator.validate_fileset(self._df["fileset"])
            cancer = self._validator.validate_cancer(self._df["cancer"])
            self._validation_mask = pd.concat(
                [
                    cid,
                    pid,
                    bd,
                    side,
                    view,
                    aid,
                    at,
                    ct,
                    cd,
                    ms,
                    mm,
                    asmt,
                    path,
                    sub,
                    fs,
                    cancer,
                ],
                axis=1,
            )
            self._validation_mask.columns = [
                "case_id",
                "patient_id",
                "breast_density",
                "left_or_right_breast",
                "image_view",
                "abnormality_id",
                "abnormality_type",
                "calc_type",
                "calc_distribution",
                "mass_shape",
                "mass_margins",
                "assessment",
                "pathology",
                "subtlety",
                "fileset",
                "cancer",
            ]

        return self._validation_mask
