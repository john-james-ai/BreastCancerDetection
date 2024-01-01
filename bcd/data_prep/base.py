#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data_prep/base.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday September 23rd 2023 12:45:39 am                                            #
# Modified   : Sunday December 31st 2023 08:15:51 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #

"""Base module for Data Preparation"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class DataPrep(ABC):
    """Defines the interface for metadata Data Preparation classes."""

    @abstractmethod
    def prep(
        self, force: bool = False, result: bool = False, **kwargs
    ) -> Union[None, pd.DataFrame]:
        """Performs the data preparation task."""
