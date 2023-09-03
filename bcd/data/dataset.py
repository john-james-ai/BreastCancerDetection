#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /bcd/data/dataset.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 31st 2023 07:36:47 pm                                               #
# Modified   : Friday September 1st 2023 03:58:21 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# %%
import pandas as pd
from studioai.data.dataset import Dataset


# ------------------------------------------------------------------------------------------------ #
class CBISMeta(Dataset):
    def summary(self) -> pd.DataFrame:
        counts = []
        cols = self._df.columns
        for col in cols:
            d = {}
            d[col] = self._df[col].value_counts()
            counts.append(d)
        df = pd.DataFrame(counts)
        return df
