#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/method/convert.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 09:59:41 pm                                                #
# Modified   : Saturday November 4th 2023 05:15:56 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image converter method module."""
import numpy as np

from bcd.preprocess.image.flow.state import Stage
from bcd.preprocess.image.method.basemethod import Method, Param


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
class Converter(Method):
    """Converts a 16-Bit image to 8-bit representation."""

    name = __qualname__
    stage = Stage(uid=0)

    @classmethod
    def execute(cls, image: np.array, params: Param = None) -> np.array:
        # Convert to float to avoid overflow or underflow.
        image = image.astype(float)
        # Rescale to gray scale values between 0-255
        img_gray = (image - image.min()) / (image.max() - image.min()) * 255.0
        # Convert to uint
        img_gray = np.uint8(img_gray)
        return img_gray
