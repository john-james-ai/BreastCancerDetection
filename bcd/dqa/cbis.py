#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dqa/cbis.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:51 am                                              #
# Modified   : Wednesday January 10th 2024 06:20:10 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Data Quality Module"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

from bcd.dqa.base import DQA

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class Validator:
    """Encapsulates methods that evaluate data validity for data quality analysis."""

    def validate_mmg_id(self, data: pd.DataFrame) -> pd.Series:
        """Evaluates validity of mmg_id data

        Args:
            df (pd.DataFrame): DataFrame containing the CBIS-DDSM data.

        Returns: Boolean mask in Series format.

        """

        mmg_ids = (
            data["abnormality_type"].apply(lambda x: x[0:4].capitalize())
            + "-"
            + data["fileset"].apply(lambda x: x.capitalize())
            + "_"
            + data["patient_id"]
            + "_"
            + data["laterality"]
            + "_"
            + data["image_view"]
        )
        return mmg_ids == data["mmg_id"]

    def validate_patient_id(self, patient_id: pd.Series) -> pd.Series:
        """Evaluates validity of patient_id data

        Args:
            patient_id (pd.Series): A series containing patient identifiers.

        Returns: Boolean mask in Series format.

        """
        return patient_id.str.contains("P_", regex=False)

    def validate_breast_density(self, breast_density: pd.Series) -> pd.Series:
        """Evaluates validity of breast_density data using the BI-RADS standard.

        Args:
            breast_density (pd.Series): A series containing breast density data.

        Returns: Boolean mask in Series format.

        """
        return breast_density.astype("int32").between(left=1, right=4)

    def validate_laterality(self, laterality: pd.Series) -> pd.Series:
        """Evaluates validity of left or right laterality data.

        Args:
            laterality (pd.Series): A series containing left or right laterality breast data.

        Returns: Boolean mask in Series format.

        """
        values = ["RIGHT", "LEFT"]
        return laterality.isin(values)

    def validate_image_view(self, image_view: pd.Series) -> pd.Series:
        """Evaluates validity of image_view data.

        Valid values are CC (cranial caudal) and MLO (mediolateral oblique)

        Args:
            image_view (pd.Series): A series containing image image_view data.

        Returns: Boolean mask in Series format.

        """
        values = ["MLO", "CC"]
        return image_view.isin(values)

    def validate_between(self, data: pd.Series, left: int, right: int) -> pd.Series:
        """Evaluates whether the data values are within a given range.

        Args:
            data (pd.Series): A series numeric data.

        Returns: Boolean mask in Series format.

        """
        return data.astype("int32").between(left=left, right=right)

    def validate_abnormality_type(self, abnormality_type: pd.Series) -> pd.Series:
        """Evaluates validity of abnormality type

        Args:
            abnormality_type (pd.Series): Validates 'calcification', and 'mass' abnormality types

        Returns: Boolean mask in Series format.

        """
        values = ["calcification", "mass"]
        return abnormality_type.isin(values)

    def validate_assessment(self, assessment: pd.Series) -> pd.Series:
        """Evaluates validity of BI-RADS assessment values.

        Args:
            assessment (pd.Series): Validates BI-RADS assessment data

        Returns: Boolean mask in Series format.

        """
        return assessment.astype("int32").between(left=0, right=6)

    def validate_pathology(self, pathology: pd.Series) -> pd.Series:
        """Evaluates validity of pathology data.

        Args:
            pathology (pd.Series): Validates pathology is in
                ['BENIGN', 'MALIGNANT', 'BENIGN_WITHOUT_CALLBACK']

        Returns: Boolean mask in Series format.

        """
        values = ["BENIGN", "MALIGNANT", "BENIGN_WITHOUT_CALLBACK"]
        return pathology.isin(values)

    def validate_subtlety(self, subtlety: pd.Series) -> pd.Series:
        """Evaluates validity of subtlety assessment values.

        Args:
            subtlety (pd.Series): Validates subtlety assessment data

        Returns: Boolean mask in Series format.

        """
        return subtlety.astype("int32").between(left=1, right=5)

    def validate_fileset(self, fileset: pd.Series) -> pd.Series:
        """Evaluates validity of fileset values

        Args:
            fileset (pd.Series): Validates fileset is in ['training', 'test']

        Returns: Boolean mask in Series format.

        """
        values = ["training", "test"]
        return fileset.isin(values)

    def validate_calc_type(self, calc_type: pd.Series) -> pd.Series:
        """Evaluates validity of calc_type values

        Args:
            calc_type (pd.Series): Calcification type values.

        Returns: Boolean mask in Series format.

        """
        values = [
            "AMORPHOUS",
            "PLEOMORPHIC",
            "ROUND_AND_REGULAR-LUCENT_CENTERED-DYSTROPHIC",
            "PUNCTATE",
            "COARSE",
            "VASCULAR",
            "FINE_LINEAR_BRANCHING",
            "LARGE_RODLIKE",
            "PUNCTATE-LUCENT_CENTERED",
            "VASCULAR-COARSE-LUCENT_CENTERED-ROUND_AND_REGULAR-PUNCTATE",
            "ROUND_AND_REGULAR-EGGSHELL",
            "PUNCTATE-PLEOMORPHIC",
            "PLEOMORPHIC-FINE_LINEAR_BRANCHING",
            "DYSTROPHIC",
            "LUCENT_CENTERED",
            "AMORPHOUS-PLEOMORPHIC",
            "ROUND_AND_REGULAR",
            "VASCULAR-COARSE-LUCENT_CENTERED",
            "COARSE-ROUND_AND_REGULAR",
            "COARSE-PLEOMORPHIC",
            "LUCENT_CENTERED",
            "VASCULAR-COARSE",
            "ROUND_AND_REGULAR-PUNCTATE",
            "ROUND_AND_REGULAR-LUCENT_CENTERED",
            "COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED",
            "SKIN",
            "LUCENT_CENTERED-PUNCTATE",
            "SKIN-PUNCTATE",
            "SKIN-PUNCTATE-ROUND_AND_REGULAR",
            "MILK_OF_CALCIUM",
            "SKIN-COARSE-ROUND_AND_REGULAR",
            "AMORPHOUS-ROUND_AND_REGULAR",
            "ROUND_AND_REGULAR-PLEOMORPHIC",
            "ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS",
            "ROUND_AND_REGULAR-AMORPHOUS",
            "COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED",
            "LARGE_RODLIKE-ROUND_AND_REGULAR",
            "ROUND_AND_REGULAR-LUCENT_CENTERED-PUNCTATE",
            "COARSE-LUCENT_CENTERED",
            "PUNCTATE-AMORPHOUS",
            "ROUND_AND_REGULAR-LUCENT_CENTERED",
            "PUNCTATE-ROUND_AND_REGULAR",
            "EGGSHELL",
            "PUNCTATE-FINE_LINEAR_BRANCHING",
            "PLEOMORPHIC-AMORPHOUS",
            "PUNCTATE-AMORPHOUS-PLEOMORPHIC",
            "NOT APPLICABLE",
        ]

        return calc_type.isin(values)

    def validate_calc_distribution(self, calc_distribution: pd.Series) -> pd.Series:
        """Evaluates validity of calc_distribution values

        Args:
            calc_distribution (pd.Series): Calcification distribution values.

        Returns: Boolean mask in Series format.

        """
        values = [
            "CLUSTERED",
            "LINEAR",
            "REGIONAL",
            "DIFFUSELY_SCATTERED",
            "SEGMENTAL",
            "CLUSTERED-LINEAR",
            "CLUSTERED-SEGMENTAL",
            "LINEAR-SEGMENTAL",
            "REGIONAL-REGIONAL",
            "NOT APPLICABLE",
        ]
        return calc_distribution.isin(values)

    def validate_mass_shape(self, mass_shape: pd.Series) -> pd.Series:
        """Evaluates validity of mass_shape values

        Args:
            mass_shape (pd.Series): Mass shape values.

        Returns: Boolean mask in Series format.

        """
        values = [
            "IRREGULAR-ARCHITECTURAL_DISTORTION",
            "ARCHITECTURAL_DISTORTION",
            "OVAL",
            "IRREGULAR",
            "LYMPH_NODE",
            "LOBULATED-LYMPH_NODE",
            "LOBULATED",
            "FOCAL_ASYMMETRIC_DENSITY",
            "ROUND",
            "LOBULATED-ARCHITECTURAL_DISTORTION",
            "ASYMMETRIC_BREAST_TISSUE",
            "LOBULATED-IRREGULAR",
            "OVAL-LYMPH_NODE",
            "LOBULATED-OVAL",
            "ROUND-OVAL",
            "IRREGULAR-FOCAL_ASYMMETRIC_DENSITY",
            "ROUND-IRREGULAR-ARCHITECTURAL_DISTORTION",
            "ROUND-LOBULATED",
            "OVAL-LOBULATED",
            "IRREGULAR-ASYMMETRIC_BREAST_TISSUE",
            "NOT APPLICABLE",
        ]

        return mass_shape.isin(values)

    def validate_mass_margins(self, mass_margins: pd.Series) -> pd.Series:
        """Evaluates validity of mass_margins values

        Args:
            mass_margins (pd.Series): Mass margin values.

        Returns: Boolean mask in Series format.

        """
        values = [
            "SPICULATED",
            "ILL_DEFINED",
            "CIRCUMSCRIBED",
            "ILL_DEFINED-SPICULATED",
            "OBSCURED",
            "OBSCURED-ILL_DEFINED",
            "MICROLOBULATED",
            "MICROLOBULATED-ILL_DEFINED-SPICULATED",
            "MICROLOBULATED-SPICULATED",
            "CIRCUMSCRIBED-ILL_DEFINED",
            "MICROLOBULATED-ILL_DEFINED",
            "CIRCUMSCRIBED-OBSCURED",
            "OBSCURED-SPICULATED",
            "OBSCURED-ILL_DEFINED-SPICULATED",
            "CIRCUMSCRIBED-MICROLOBULATED",
            "OBSCURED-CIRCUMSCRIBED",
            "CIRCUMSCRIBED-SPICULATED",
            "CIRCUMSCRIBED-OBSCURED-ILL_DEFINED",
            "CIRCUMSCRIBED-MICROLOBULATED-ILL_DEFINED",
            "NOT APPLICABLE",
        ]

        return mass_margins.isin(values)

    def validate_cancer(self, cancer: pd.Series) -> pd.Series:
        """Evaluates the validity of cancer values.

        Args:
            cancer (pd.Series): Series containing cancer variable.

         Returns: Boolean mask in Series format.

        """
        values = [True, False]
        return cancer.isin(values)

    def validate_filepath(self, filepath: pd.Series) -> pd.Series:
        """Evaluates validity and existence of filepaths

        Args:
            filepaths (pd.Series): Validates filepaths

        Returns: Boolean mask in Series format.

        """
        # pylint: disable=unnecessary-lambda
        return filepath.apply(lambda x: os.path.exists(x))

    def validate_bit_depth(self, bit_depth: pd.Series) -> pd.Series:
        """Evaluates validity of bit_depth

        Args:
            bit_depth (pd.Series): Validates bit_depth

        Returns: Boolean mask in Series format.

        """
        values = [8, 16]
        return bit_depth.astype("int32").isin(values)

    def validate_greater_equal(self, values: pd.Series, minval: int) -> pd.Series:
        """Evaluates whether a series of values is greater than or equal to minval.

        This is used to validate that the number of rows and columns in the
        images is greater than or equal to 256, the size of images that
        will be fed to the detection and classification model.

        Args:
            values (pd.Series): Values to be evaluated
            minval (int): The minimum value

        Returns: Boolean mask in Series format.

        """
        return values >= minval

    def validate_aspect_ratio(self, aspect_ratio: pd.Series) -> pd.Series:
        """Evaluates validity of aspect_ratio

        Args:
            aspect_ratio (pd.Series): Series containing aspect_ratio

        Returns: Boolean mask in Series format.

        """
        return aspect_ratio.between(left=0, right=1)

    def validate_size(self, size: pd.Series) -> pd.Series:
        """Evaluates validity of size

        Ensures the size is greater or equal to 256**2

        Args:
            size (pd.Series): Series containing pixel array size.

        Returns: Boolean mask in Series format.

        """
        return size >= 256**2

    def validate_file_size(self, file_size: pd.Series) -> pd.Series:
        """Validates filesize is greater 1 KB

        Args:
            file_size (pd.Series): File size in bytes
        """
        return file_size > 1000

    def validate_min_pixel_value(self, df: pd.DataFrame) -> pd.Series:
        """Validates the minimum pixel value

        Args:
            df (pd.DataFrame): DataFrame containing the CBIS Data
        """
        return df["min_pixel_value"] < df["max_pixel_value"]

    def validate_max_pixel_value(self, df: pd.DataFrame) -> pd.Series:
        """Evaluates a pixel statistic

        Args:
            df (pd.DataFrame): DataFrame containing the CBIS Data
        """
        max_value = 2 ** df["bit_depth"]
        return df["max_pixel_value"] <= max_value

    def validate_mean_pixel_value(self, df: pd.DataFrame) -> pd.Series:
        """Validates the mean pixel value

        Args:
            df (pd.DataFrame): DataFrame containing the CBIS Data
        """
        return (0 < df["mean_pixel_value"]) & (
            df["mean_pixel_value"] < 2 ** df["bit_depth"]
        )

    def validate_std_pixel_value(self, std_pixel_value: pd.Series) -> pd.Series:
        """Validates the std pixel value

        Args:
            std_pixel_value (pd.Series): Series containing std pixel value
        """
        return 0 < std_pixel_value


# ------------------------------------------------------------------------------------------------ #
class CBISDQA(DQA):
    """Enapsulates the Case data quality analysis"""

    # Keys used in the FileManager object to obtain the associated filenames.
    __NAME = "CBIS-DDSM"

    def __init__(
        self,
        data: pd.DataFrame,
        validator: type[Validator] = Validator,
    ) -> None:
        super().__init__(data=data, name=self.__NAME)

        self._validator = validator()
        self._validation_mask = None

    def validate(self) -> np.ndarray:
        "Validates the data and returns a boolean mask of cell validity."
        if self._validation_mask is None:
            mid = self._validator.validate_mmg_id(self._df)
            pid = self._validator.validate_patient_id(self._df["patient_id"])
            bd = self._validator.validate_breast_density(self._df["breast_density"])
            laterality = self._validator.validate_laterality(self._df["laterality"])
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
            bit_depth = self._validator.validate_bit_depth(self._df["bit_depth"])
            nrows = self._validator.validate_greater_equal(self._df["rows"], minval=256)
            ncols = self._validator.validate_greater_equal(self._df["cols"], minval=256)
            ar = self._validator.validate_aspect_ratio(self._df["aspect_ratio"])
            size = self._validator.validate_size(self._df["size"])
            file_size = self._validator.validate_file_size(self._df["file_size"])
            min_pixel_value = self._validator.validate_min_pixel_value(self._df)
            max_pixel_value = self._validator.validate_max_pixel_value(self._df)
            mean_pixel_value = self._validator.validate_mean_pixel_value(self._df)
            std_pixel_value = self._validator.validate_std_pixel_value(
                self._df["std_pixel_value"]
            )
            filepath = self._validator.validate_filepath(self._df["filepath"])
            self._validation_mask = pd.concat(
                [
                    mid,
                    pid,
                    bd,
                    laterality,
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
                    bit_depth,
                    nrows,
                    ncols,
                    ar,
                    size,
                    file_size,
                    min_pixel_value,
                    max_pixel_value,
                    mean_pixel_value,
                    std_pixel_value,
                    filepath,
                ],
                axis=1,
            )
            self._validation_mask.columns = [
                "mmg_id",
                "patient_id",
                "breast_density",
                "laterality",
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
                "bit_depth",
                "rows",
                "cols",
                "aspect_ratio",
                "size",
                "file_size",
                "min_pixel_value",
                "max_pixel_value",
                "mean_pixel_value",
                "std_pixel_value",
                "filepath",
            ]

        return self._validation_mask
