#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/case/dqa.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:51 am                                              #
# Modified   : Saturday September 23rd 2023 03:31:25 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Data Quality Module"""
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
class CalcCaseDQA(DQA):
    def __init__(self, filepath: str, validator: Validator = Validator, name: str = None) -> None:
        super().__init__(filepath=filepath, name=name)
        self._validator = validator()
        self._df = pd.read_csv(self._filepath)
        self._validation_mask = None

    def validate(self) -> np.ndarray:
        "Validates the data and returns a boolean mask of cell validity."
        if self._validation_mask is None:
            pid = self._validator.validate_patient_id(self._df["patient_id"])
            bd = self._validator.validate_breast_density(self._df["breast_density"])
            side = self._validator.validate_side(self._df["left_or_right_breast"])
            view = self._validator.validate_image_view(self._df["image_view"])
            aid = self._validator.validate_between(self._df["abnormality_id"], left=1, right=10)
            at = self._validator.validate_abnormality_type(self._df["abnormality_type"])
            asmt = self._validator.validate_assessment(self._df["assessment"])
            ct = self._validator.validate_calc_type(self._df["calc_type"])
            cd = self._validator.validate_calc_distribution(self._df["calc_distribution"])
            path = self._validator.validate_pathology(self._df["pathology"])
            sub = self._validator.validate_subtlety(self._df["subtlety"])
            ds = self._validator.validate_dataset(self._df["dataset"])
            isuid = self._validator.validate_series_uid(series_uid=self._df["image_series_uid"])
            roisuid = self._validator.validate_series_uid(
                series_uid=self._df["roi_mask_series_uid"]
            )
            cropsuid = self._validator.validate_series_uid(
                series_uid=self._df["cropped_image_series_uid"]
            )
            self._validation_mask = pd.concat(
                [
                    pid,
                    bd,
                    side,
                    view,
                    aid,
                    at,
                    ct,
                    cd,
                    asmt,
                    path,
                    sub,
                    ds,
                    isuid,
                    roisuid,
                    cropsuid,
                ],
                axis=1,
            )
            self._validation_mask.columns = [
                "patient_id",
                "breast_density",
                "left_or_right_breast",
                "image_view",
                "abnormality_id",
                "abnormality_type",
                "calc_type",
                "calc_distribution",
                "assessment",
                "pathology",
                "subtlety",
                "dataset",
                "image_series_uid",
                "roi_mask_series_uid",
                "cropped_image_series_uid",
            ]

        return self._validation_mask


# ------------------------------------------------------------------------------------------------ #
class MassCaseDQA(DQA):
    def __init__(self, filepath: str, validator: Validator = Validator, name: str = None) -> None:
        super().__init__(filepath=filepath, name=name)
        self._validator = validator()
        self._df = pd.read_csv(self._filepath)
        self._validation_mask = None

    def validate(self) -> np.ndarray:
        "Validates the data and returns a boolean mask of cell validity."
        if self._validation_mask is None:
            pid = self._validator.validate_patient_id(self._df["patient_id"])
            bd = self._validator.validate_breast_density(self._df["breast_density"])
            side = self._validator.validate_side(self._df["left_or_right_breast"])
            view = self._validator.validate_image_view(self._df["image_view"])
            aid = self._validator.validate_between(self._df["abnormality_id"], left=1, right=10)
            at = self._validator.validate_abnormality_type(self._df["abnormality_type"])
            asmt = self._validator.validate_assessment(self._df["assessment"])
            ms = self._validator.validate_mass_shape(self._df["mass_shape"])
            mm = self._validator.validate_mass_margins(self._df["mass_margins"])
            path = self._validator.validate_pathology(self._df["pathology"])
            sub = self._validator.validate_subtlety(self._df["subtlety"])
            ds = self._validator.validate_dataset(self._df["dataset"])
            isuid = self._validator.validate_series_uid(series_uid=self._df["image_series_uid"])
            roisuid = self._validator.validate_series_uid(
                series_uid=self._df["roi_mask_series_uid"]
            )
            cropsuid = self._validator.validate_series_uid(
                series_uid=self._df["cropped_image_series_uid"]
            )
            self._validation_mask = pd.concat(
                [
                    pid,
                    bd,
                    side,
                    view,
                    aid,
                    at,
                    ms,
                    mm,
                    asmt,
                    path,
                    sub,
                    ds,
                    isuid,
                    roisuid,
                    cropsuid,
                ],
                axis=1,
            )
            self._validation_mask.columns = [
                "patient_id",
                "breast_density",
                "left_or_right_breast",
                "image_view",
                "abnormality_id",
                "abnormality_type",
                "mass_shape",
                "mass_margins",
                "assessment",
                "pathology",
                "subtlety",
                "dataset",
                "image_series_uid",
                "roi_mask_series_uid",
                "cropped_image_series_uid",
            ]

        return self._validation_mask
