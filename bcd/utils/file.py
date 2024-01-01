#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.8                                                                              #
# Filename   : /bcd/utils/file.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday April 4th 2023 08:46:04 pm                                                  #
# Modified   : Sunday December 31st 2023 05:18:24 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os


# ------------------------------------------------------------------------------------------------ #
def getsize(filepath: str) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = os.stat(filepath).st_size
    idx = 0
    while size > 1000:
        idx += 1
        size /= 1024
    return f"{round(size,2)} {units[idx]}"
