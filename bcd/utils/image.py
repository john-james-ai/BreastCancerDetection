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
# Modified   : Tuesday December 26th 2023 03:42:04 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Images Utilities"""
from typing import Union

import cv2
import numpy as np


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
#                                  PIXEL RANGE CONVERTER                                           #
# ------------------------------------------------------------------------------------------------ #
def convert_pixel_range(
    image: np.ndarray, from_values: tuple, to_values: tuple
) -> np.ndarray:
    """Changes pixel values from from_range to to_range

    Args:
        image (np.ndarray): Image in numpy array format.
        from_values (tuple): Tuple in form (min_pixel_value,  max_pixel_value) indicating the
            pixel values in the image.
        to_values (tuple): Tuple in form (min_pixel_value,  max_pixel_value) indicating
            the pixel values to which the image is to be converted.

    Returns:
        np.ndarrray: The converted image.

    """
    from_range = from_values[1] - from_values[0]
    to_range = to_values[1] - to_values[0]
    scaled = np.array((image - from_values[0]) / float(from_range), dtype=float)
    return to_values[0] + (scaled * to_range)


# ------------------------------------------------------------------------------------------------ #
#                                       GRAYSCALE                                                  #
# ------------------------------------------------------------------------------------------------ #
def grayscale(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Returns a 2-dimensional grayscale image with pixel values in 0 to 255.

    This method performs three tasks:
        1. If the image is in 3-d, it is converted to a 2-d image.
        2. If the range of pixels is converted to [0,255], if not in that range.
        3. Image is converted to uint8 format.

    It also has an invert option to invert the image intensity values.

    Args:
        image (np.ndarray): Image with 2d or 3d shape
        invert (bool): Whether to invert the
    """
    if len(image.shape) > 2:
        image = image[:, :, 0].squeeze()
    # Converts range to [0,255] if needed.
    if np.max(image) == 1.0:
        image = convert_pixel_range(image=image, from_values=(0, 1), to_values=(0, 255))
    elif np.max(image) > 255:
        image = convert_pixel_range(
            image=image, from_values=(0, np.max(image)), to_values=(0, 255)
        )
    if invert:
        image = 255 - image
    return image.astype("uint8")


# ------------------------------------------------------------------------------------------------ #
#                                         ORIENT                                                   #
# ------------------------------------------------------------------------------------------------ #
def orient_image(image: np.ndarray, left: bool = True) -> Union[np.ndarray, bool]:
    """Orients a mammogram to left or right side.

    Counts the nonzero pixel values in the left and right 10% of the image. If the
    left 10%  has a greater number of nonzero pixels than the right 10%, the image is
    oriented left. Otherwise, it is oriented right. The image is flipped based upon the
    desired orientation indicated by the 'left' parameter.

    Args:
        image (np.ndarray): Mammogram image in numpy format.
        left (bool): Orient left if True, otherwise, orient right. Default is True.

    Returns
        np.ndarray: Image oriented
        bool: Whether the image was flipped.
    """
    flipped = False
    left_nonzero = cv2.countNonZero(image[:, 0 : int(image.shape[1] * 0.10)][0])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] * 0.90) :][0])

    if (left_nonzero < right_nonzero) and left:
        image = cv2.flip(image, 1)
        flipped = True
    elif (left_nonzero > right_nonzero) and not left:
        image = cv2.flip(image, 1)
        flipped = True
    return image, flipped
