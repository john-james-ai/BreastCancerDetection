#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/dataset.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday January 1st 2024 04:52:28 am                                                 #
# Modified   : Monday January 1st 2024 05:10:17 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from bcd.analyze.dqa.base import DQA
from bcd.dal.file import IOService


class Dataset(ABC):
    """Encapsulates CBIS-DDSM data and analysis, visualization, and experimentation capabilities.

    Args:
        case_filepath (str): Filepath to the case data
        series_filepath (str): Filepath to image series data
    """

    def __init__(self, case_filepath: str, series_filepath: str) -> None:
        self._case_filepath = case_filepath
        self._series_filepath = series_filepath
        self._case = None
        self._series = None
        self.load()

    def add_dqa(self, dqa: DQA) -> None:
        """Adds a data quality module to the Dataset

        Args:
            dqa (DQA): Data Quality Module.
        """

    def load(self) -> None:
        """Loads case and series data into the Dataset"""
        self._case = IOService.read(self._case_filepath)
        self._series = IOService.read(self._series_filepath)

    def save(self) -> None:
        """Saves case and series data to file."""
        self._case.to_csv(self._case_filepath, index=False)
        self._series.to_csv(self._series_filepath, index=False)
