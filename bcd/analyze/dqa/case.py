#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/analyze/dqa/case.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:51 am                                              #
# Modified   : Friday December 29th 2023 02:15:18 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Data Quality Module"""
import logging
import sys

import numpy as np
import pandas as pd

from bcd.analyze.dqa.base import DQA, Validator
from bcd.data.base import Dataset
from bcd.utils.file import IOService

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class CaseDQA(DQA):
    """Enapsulates the Case data quality analysis"""

    # Keys used in the FileManager object to obtain the associated filenames.
    __DATASETS = ["mass_train", "mass_test", "calc_train", "calc_test"]
    __MORPHOLOGY = ["calc type", "calc distribution", "mass shape", "mass margins"]
    __NAME = "Case"

    def __init__(
        self,
        dataset: Dataset,
        validator: Validator = Validator,
        io: IOService = IOService,
    ) -> None:
        super().__init__(name=self.__NAME)
        self._file_manager = file_manager
        self._validator = validator()
        self._io = io
        self._df = None
        self._validation_mask = None
        self.load_data()

    def load_data(self) -> None:
        """Loads the case data into a single data frame for analysis."""
        case_datasets = []
        for dataset in self.__DATASETS:
            filepath = self._file_manager.get_raw_metadata_filepath(name=dataset)
            df = self._io.read(filepath=filepath)
            df = self._add_fileset(name=dataset, df=df)
            case_datasets.append(df)
        df = pd.concat(case_datasets, axis=0, join="outer")

        # Calcification and mass morphologies are added and NaN must be
        # changed to 'Not Applicable' as appropriate.
        df[self.__MORPHOLOGY] = df[self.__MORPHOLOGY].fillna("Not Applicable")
        logger.debug(df.head())
        self._df = df

    def validate(self) -> np.ndarray:
        "Validates the data and returns a boolean mask of cell validity."
        if self._validation_mask is None:
            cid = self._validator.validate_case_id(self._df)
            pid = self._validator.validate_patient_id(self._df["patient_id"])
            bd = self._validator.validate_breast_density(self._df["breast_density"])
            side = self._validator.validate_side(self._df["left_or_right_breast"])
            image_view = self._validator.validate_image_view(self._df["image_view"])
            aid = self._validator.validate_between(
                self._df["abnormality_id"], left=1, right=10
            )
            at = self._validator.validate_abnormality_type(self._df["abnormality_type"])
            ct = self._validator.validate_calc_type(self._df["calc_type"])
            cd = self._validator.validate_calc_distribution(
                self._df["calc_distribution"]
            )
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
                    image_view,
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

    def _add_fileset(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Adds the fileset variable to the dataframe."""
        fileset = name.split("_")[1]
        df["fileset"] = fileset
        return df
