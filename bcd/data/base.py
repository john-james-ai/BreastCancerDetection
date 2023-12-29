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
# Modified   : Friday December 29th 2023 02:15:17 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base Module for the Data Package"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from bcd import DataClass
from bcd.analyze.dqa.base import DQA
from bcd.dal.base import Database
from bcd.explore.meta import Explorer


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetDTO(DataClass):
    """Data transfer object for the Dataset."""

    name: str
    description: str
    dataset_type: str
    stage: str
    dataset_format: str
    data: Any

    @classmethod
    def from_dict(cls, ds_dict: dict) -> DatasetDTO:
        return cls(
            name=ds_dict["name"],
            description=ds_dict["description"],
            dataset_type=ds_dict["dataset_type"],
            stage=ds_dict["stage"],
            dataset_format=ds_dict["dataset_format"],
            data=ds_dict["data"],
        )


# ------------------------------------------------------------------------------------------------ #
class Dataset(ABC):
    """Encapsulates a collection of Data in tabular form."""

    def __init__(
        self,
        database: Database,
        name: str,
        description: str,
        dataset_type: str = "metadata",
        stage: str = "exp",
        dataset_format: str = "csv",
        data: Any = None,
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

        self._dqa = None
        self._eda = None

    @property
    def data(self) -> Any:
        return self._data

    @property
    def dqa(self) -> DQA:
        return self._dqa

    @dqa.setter
    def dqa(self, dqa: DQA) -> None:
        self._dqa = dqa

    @abstractmethod
    def summarize(self) -> pd.DataFrame:
        """Summarizes the dataset"""

    def load(self) -> None:
        """Loads the data into the Dataset object"""
        self._data = self._database.read(name=self._name)

    def save(self) -> None:
        """Saves the Dataset's data to the database."""
        self._database.save(self.as_dto)

    def as_dto(self) -> dict:
        """Returns the Dataset as a dictionary"""
        d = {
            "name": self._name,
            "description": self._description,
            "dataset_type": self._dataset_type,
            "stage": self._stage,
            "dataset_format": self._dataset_format,
            "data": self._data,
        }
        return DatasetDTO.from_dict(ds_dict=d)
