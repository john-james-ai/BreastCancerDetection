#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/task.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday November 6th 2023 04:33:38 am                                                #
# Modified   : Monday November 6th 2023 06:12:52 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from bcd import DataClass
from bcd.utils.date import to_datetime

# ------------------------------------------------------------------------------------------------ #


@dataclass
class Task(DataClass):
    """Encapsulates a Task defined as a method and its parameters."""

    uid: str
    mode: str
    stage_id: int
    stage: str
    method: str
    params: dict
    images_processed: int = 0
    created: datetime = None

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Task:
        return cls(
            uid=df["uid"].values[0],
            mode=df["mode"].values[0],
            stage_id=df["stage_id"].values[0],
            stage=df["stage"].values[0],
            method=df["method"].values[0],
            params=df["params"].values[0],
            images_processed=df["images_processed"].values[0],
            created=to_datetime(df["created"].values[0]),
        )
