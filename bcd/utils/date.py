#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/utils/date.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 27th 2023 02:23:44 am                                                #
# Modified   : Friday October 27th 2023 02:26:46 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Date Utilities"""
from datetime import datetime

import numpy as np
# ------------------------------------------------------------------------------------------------ #

def to_datetime(dt: np.datetime64) -> datetime:
    if dt is None:
        return None
    else:
        timestamp = ((dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1,'s'))
        return datetime.utcfromtimestamp(timestamp)