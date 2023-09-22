#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/series/dataset.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:24:16 am                                              #
# Modified   : Friday September 22nd 2023 03:34:52 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Series Dataset Module"""
import sys
import os
import logging

import pandas as pd

from bcd.data.base import CBISDataset

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------------------------ #
SERIES_DTYPES = {
    "series_uid": "str",
    "series_description": "category",
    "number_of_images": "int32",
    "file_location": "str",
}
SERIES_COLS = SERIES_DTYPES.keys()


# ------------------------------------------------------------------------------------------------ #
class SeriesDataset(CBISDataset):
    """Dataset containing series metadata

    Args:
        filepath (str): File path to the series metadata.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = os.path.abspath(filepath)
        df = pd.read_csv(self._filepath, dtype=SERIES_DTYPES)
        super().__init__(df=df)
