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
# Modified   : Monday December 11th 2023 12:41:04 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Images Utilities"""
import numpy as np
from skimage.util import random_noise

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
    img = img.astype(float)
    img = np.array(255 * img, dtype="uint8")
    if invert:
        img = 255 - img
    if asfloat:
        img = np.asfarray(img)
    return img


# ------------------------------------------------------------------------------------------------ #
#                                       NOISER                                                     #
# ------------------------------------------------------------------------------------------------ #
class Noiser:
    """Adds random noise to an image"""

    def __init__(
        self,
        mean: float = 0,
        var_gaussian: float = 0.05,
        var_speckle: float = 0.01,
        amount: float = 0.05,
        svp: float = 0.5,
    ) -> None:
        self._mean = mean
        self._var_gaussian = var_gaussian
        self._var_speckle = var_speckle
        self._amount = amount
        self._svp = svp

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        image = random_noise(
            image=image, mode="gaussian", mean=self._mean, var=self._var_gaussian
        )
        image = random_noise(
            image=image, mode="speckle", mean=self._mean, var=self._var_speckle
        )
        image = random_noise(image=image, mode="poisson")
        image = random_noise(
            image=image, mode="s&p", amount=self._amount, salt_vs_pepper=self._svp
        )
        return convert_uint8(image)
