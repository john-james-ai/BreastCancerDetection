#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/series/prep.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 03:24:16 am                                              #
# Modified   : Friday September 22nd 2023 04:54:10 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Series Module"""
import sys
import os
import logging
from typing import Union

import pandas as pd

from bcd.data.base import DataPrep

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
class SeriesPrep(DataPrep):
    def prep(
        self,
        fp_in: str,
        fp_out: str,
        force: bool = False,
        result: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """Extracts relevant data from the series metadata file.

        Args:
            fp_in (str): Input file path for the series metadata
            fp_out (str): Output file path for the series metadata
            force (bool): Whether to force execution if output already exists. Default is False.
            result (bool): Whether the result should be returned. Default is False.
        """

        fp_in = os.path.abspath(fp_in)
        fp_out = os.path.abspath(fp_out)

        os.makedirs(os.path.dirname(fp_out), exist_ok=True)

        if force or not os.path.exists(fp_out):
            df1 = pd.read_csv(fp_in)
            df2 = df1[SERIES_COLS].copy()
            df2 = self._format_column_names(df=df2)
            df2["file_location"] = df2["file_location"].str.replace(
                "./CBIS-DDSM", "data/raw/CBIS-DDSM", regex=False
            )
            df2.to_csv(fp_out, index=False)
        if result:
            return pd.read_csv(fp_out)
