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
# Modified   : Saturday December 30th 2023 08:51:57 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base Module for the Data Package"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from bcd.analyze.dqa.base import DQA
from bcd.dal.base import Database
from bcd.explore.meta import Explorer


# ------------------------------------------------------------------------------------------------ #
class Dataset(ABC):
    """Encapsulates a collection of Data in tabular form."""

    def __init__(
        self,
        name: str,
        description: str,
        dataset_type: str = "metadata",
        stage: str = "exp",
        dataset_format: str = "csv",
        data: Any = None,
        database: type[Database] = Database,
    ) -> None:
        self._database = database
        self._name = name
        self._description = description
        self._dataset_type = dataset_type
        self._stage = stage
        self._dataset_format = dataset_format
        self._data = data
        if self._data is None:
            self.load()

        self.dqa = None
        self.eda = None

    @abstractmethod
    def summarize(self) -> pd.DataFrame:
        """Summarizes the dataset"""

    @classmethod
    def create(cls, name: str) -> Dataset:
        """Factory method that creates a Dataset instance from existing data in the Database.

        Args:
            name (str): Name of the data set in the DataBase

        """
        dataset = cls._database.read(name=name)

    def add_dqa_module(self, dqa: type[DQA]) -> None:
        """Adds a Data Quality Assessment Module to the Dataset

        Args:
            dqa (type[DQA]): A DQA Class
        """
        self.dqa = dqa(data=self._data)

    def add_eda_module(self, eda: type[Explorer]) -> None:
        """Adds an Exploratory Data Analysis Module to the Dataset

        Args:
            eda (type[Explorer]): An Explorer class
        """
        self.eda = eda(data=self._data)

    def load(self) -> None:
        """Loads the data into the Dataset object"""
        self._data = self._database.read(name=self._name)

    def save(self) -> None:
        """Saves the Dataset's data to the database."""
        self._database.create(self.as_dict)

    def as_dict(self) -> dict:
        """Returns the Dataset as a dictionary"""
        return {
            "name": self._name,
            "description": self._description,
            "dataset_type": self._dataset_type,
            "stage": self._stage,
            "dataset_format": self._dataset_format,
            "data": self._data,
        }
