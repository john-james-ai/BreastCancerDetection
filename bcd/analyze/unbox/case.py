#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/analyze/unbox/case.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday December 31st 2023 12:21:09 am                                               #
# Modified   : Sunday December 31st 2023 05:17:35 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bcd.dal.file import IOService

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")
sns.set_palette("Blues_r")
# ------------------------------------------------------------------------------------------------ #


class UnboxCases:
    """This class unboxes the case training and test sets.

    First, the training and test sets are combined into a single case dataframe. Fileset
    is added as the 'train', 'test', indicator in the combined file. The combined
    file is then saved to the interim directory.

    Args:
        train_filepath (str): Path to the training file
        test_filepath (str): Path to the test file
        case_filename (str): Name of the combined case file to be stored in the
            interim directory below.
    """

    __DIRECTORY = "data/meta/1_interim"

    def __init__(
        self, train_filepath: str, test_filepath: str, case_filename: str
    ) -> None:
        self._df = self._combine_train_test(train_filepath, test_filepath)
        self._case_filename = case_filename
        self._head = None

    def save(self) -> None:
        case_filepath = os.path.join(self.__DIRECTORY, self._case_filename)
        IOService.write(filepath=case_filepath, data=self._df)

    def _combine_train_test(
        self, train_filepath: str, test_filepath: str
    ) -> pd.DataFrame:
        """Combines the training and test sets into a single dataframe with fileset indicated."""
        train = IOService.read(train_filepath)
        test = IOService.read(test_filepath)
        train["fileset"] = "train"
        test["fileset"] = "test"
        return pd.concat([train, test], axis=0)
