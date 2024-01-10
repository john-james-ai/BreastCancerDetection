#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/base.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday December 29th 2023 12:00:22 am                                               #
# Modified   : Wednesday January 10th 2024 06:20:10 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base Module for the Data Package"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from bcd.dqa.base import DQA
from bcd.explore.meta import Explorer as EDA


# ------------------------------------------------------------------------------------------------ #
class Dataset(ABC):
    """Encapsulates a collection of Data in tabular form."""

    @property
    @abstractmethod
    def dqa(self) -> DQA:
        """Access to the data quality analysis module"""

    @property
    @abstractmethod
    def eda(self) -> EDA:
        """Access to the exploratory data analysis module"""

    @abstractmethod
    def summarize(self) -> pd.DataFrame:
        """Summarizes the dataset"""
