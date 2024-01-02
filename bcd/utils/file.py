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
# Modified   : Tuesday January 2nd 2024 05:34:21 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from typing import Union


# ------------------------------------------------------------------------------------------------ #
def getsize(filepath: str, as_bytes: bool = False) -> Union[str, int]:
    """Returns the size of a file on disk.

    Args:
        filepath (str): Path to file
        as_bytes (bool): Whether to return the size as an integer number of bytes
         or as a string with units.

    """
    units = ["B", "KB", "MB", "GB"]
    size = os.stat(filepath).st_size
    if as_bytes:
        return size
    else:
        idx = 0
        while size > 1000:
            idx += 1
            size /= 1024
        return f"{round(size,2)} {units[idx]}"
