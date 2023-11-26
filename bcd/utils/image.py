#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/utils/image.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday November 18th 2023 12:29:17 pm                                             #
# Modified   : Sunday November 26th 2023 05:59:37 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Images Utilities"""
import numpy as np

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #


def convert_uint8(
    img: np.ndarray, invert: bool = False, asfloat: bool = False
) -> np.ndarray:
    """Converts floating point array in [0,1] to unit8 in [9,255]

    This is used on the output of skimage random_noise function that returns a normalized
    image with values in [0,1]. This function converts the pixel values back to that
    of an 8-bit unsigned representation with values in [0,255].

    Args:
        img (np.ndarray): The image in numpy array format.
        invert (bool): Whether to invert the colors

    """
    img = np.array(255 * img, dtype="uint8")
    if invert:
        img = 255 - img
    if asfloat:
        img = np.asfarray(img)
    return img
