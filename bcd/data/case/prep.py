#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/case/prep.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:23:38 am                                              #
# Modified   : Sunday September 24th 2023 06:51:23 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Case Data Prep Module"""
import sys
import os
import logging
from typing import Union

import pandas as pd
import numpy as np

from bcd.data.prep import DataPrep

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class CasePrep(DataPrep):
    def prep(
        self,
        train_fp: str,
        test_fp: str,
        cases_fp: str,
        force: bool = False,
        result: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """Combines training and test cases into a single csv case file.

        Args:
            train_fp (str): File path to the training set
            test_fp (str): File path to the test set
            cases_fp (str): File path to the combined cases dataset.
            force (bool): Whether to force execution if output already exists. Default is False.
            result (bool): Whether the result should be returned. Default is False.
        """
        train_fp = os.path.abspath(train_fp)
        test_fp = os.path.abspath(test_fp)
        cases_fp = os.path.abspath(cases_fp)

        os.makedirs(os.path.dirname(cases_fp), exist_ok=True)

        def extract_series_uid(filepath) -> str:
            return filepath.split("/")[2]

        if force or not os.path.exists(cases_fp):
            df1 = pd.read_csv(train_fp)
            df2 = pd.read_csv(test_fp)
            df1["dataset"] = "train"
            df2["dataset"] = "test"
            df3 = pd.concat([df1, df2], axis=0)
            df3["image_series_uid"] = df3["image file path"].apply(extract_series_uid)
            df3["roi_mask_series_uid"] = df3["ROI mask file path"].apply(extract_series_uid)
            df3["cropped_image_series_uid"] = df3["cropped image file path"].apply(
                extract_series_uid
            )
            df3.drop(
                columns=["image file path", "ROI mask file path", "cropped image file path"],
                inplace=True,
            )
            df3["pathology"] = np.where(df3["pathology"] == "MALIGNANT", "MALIGNANT", "BENIGN")
            df3 = self._format_column_names(df=df3)

            df3.to_csv(cases_fp, index=False)
        if result:
            return pd.read_csv(cases_fp)


# ------------------------------------------------------------------------------------------------ #
class MasterCasePrep(DataPrep):
    def prep(
        self, mass_fp: str, calc_fp: str, master_fp: str, force: bool = False, result: bool = False
    ) -> Union[None, pd.DataFrame]:
        """Combines mass and calcification cases and converts to long-form by series description.

        Args:
            mass_fp (str): File path to the mass dataset
            calc_fp (str): File path to the calc dataset
            master_fp (str): File path to master case dataset.
            force (bool): Whether to force execution if output already exists. Default is False.
            result (bool): Whether the result should be returned. Default is False.
        """
        mass_fp = os.path.abspath(mass_fp)
        calc_fp = os.path.abspath(calc_fp)
        master_fp = os.path.abspath(master_fp)

        os.makedirs(os.path.dirname(master_fp), exist_ok=True)

        if force or not os.path.exists(master_fp):
            dfm = pd.read_csv(mass_fp)
            dfc = pd.read_csv(calc_fp)
            df = pd.concat([dfm, dfc], axis=0)
            dfi = df.drop(columns=["roi_mask_series_uid", "cropped_image_series_uid"]).rename(
                columns={"image_series_uid": "series_uid"}
            )
            dfr = df.drop(columns=["image_series_uid", "cropped_image_series_uid"]).rename(
                columns={"roi_mask_series_uid": "series_uid"}
            )
            dfc = df.drop(columns=["roi_mask_series_uid", "image_series_uid"]).rename(
                columns={"cropped_image_series_uid": "series_uid"}
            )
            dfi["series_description"] = "full mammogram images"
            dfr["series_description"] = "ROI mask images"
            dfc["series_description"] = "cropped images"
            dfm = pd.concat([dfi, dfr, dfc], axis=0)
            dfm = dfm.drop_duplicates(subset=["series_uid"])
            dfm.to_csv(master_fp, index=False)

        if result:
            return pd.read_csv(master_fp)
