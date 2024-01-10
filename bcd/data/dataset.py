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
# Modified   : Monday January 8th 2024 09:17:05 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import pandas as pd

from bcd.analyze.dqa.cbis import CBISDQA
from bcd.dal.file import IOService
from bcd.data.base import Dataset
from bcd.explore.meta import Explorer as EDA

# ------------------------------------------------------------------------------------------------ #


class CBISDataset(Dataset):
    """Encapsulates CBIS-DDSM data and analysis, visualization, and experimentation capabilities.

    Args:
        filepath (str): Filepath to the CBIS-DDSM data
    """

    def __init__(self, filepath: str, dqa: type[CBISDQA] = CBISDQA) -> None:
        self._filepath = filepath
        self._df = None
        self._eda = None
        self._show = None
        self._load()
        self._dqa = dqa(data=self._df)

    @property
    def dqa(self) -> CBISDQA:
        """Access to the data quality analysis module"""
        return self._dqa

    @property
    def eda(self) -> EDA:
        """Access to the exploratory data analysis module"""
        return self._eda

    # @property
    # def show(self) -> EDA:
    #     """Access to the image visualizer"""
    #     return self._eda

    # @show.setter
    # def show(self, visualizer: type[Visualizer]) -> None:
    #     """Sets the image visualizer

    #     Args:
    #         show (type[Visualizer]); The Visualizer Class type

    #     """
    #     self._show = visualizer(data=self._df)

    def summarize(self) -> pd.DataFrame:
        """Summarizes the datasets variables, types, NA values, and uniqueness."""
        cols = self._df.columns
        dtypes = self._df.dtypes
        nonna = self._df.count()
        na = self._df.isna().sum(axis=0)
        completeness = nonna / self._df.shape[0]
        unique = self._df.nunique(axis=0)
        uniqueness = unique / nonna
        d = {
            "Variable": cols,
            "Data Types": dtypes,
            "Non NA": nonna,
            "NA": na,
            "Completeness": completeness,
            "Unique": unique,
            "Uniqueness": uniqueness,
        }
        df = pd.DataFrame(data=d)
        return df

    def _load(self) -> None:
        """Loads case and series data into the Dataset"""
        self._df = IOService.read(self._filepath)
