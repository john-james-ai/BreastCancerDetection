#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday December 25th 2023 03:13:51 pm                                               #
# Modified   : Thursday January 11th 2024 03:27:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging

import cv2
import numpy as np

from bcd.preprocess.base import Task
from bcd.utils.profile import profiler


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
#                                      IMAGE RESIZER                                               #
# ------------------------------------------------------------------------------------------------ #
class Resizer(Task):
    """Class resizes images using various types of interpolation and aspect ratios

    Args:
        size (tuple): Size of resulting image as a tuple, i.e., (height, width)
        keep_aspect (bool): Whether to keep the aspect ratio. If True, padding will be added
            before resizing. Default is False.
        center (bool): If keep_aspect is True, and padding is therefore added, should
            the image be centered or oriented to the top left of the resized image.
            Default is False.
        interpolation (str): Interpolation method corresponding to those supported by OpenCV
            Supported values are 'area', 'nearest', 'linear', 'cubic', 'nearest_exact', and
            'linear_exact'. Default is 'area'.
        padding_color (int): The value to use for padding, if keep_aspect is True. Default = 0.

    """

    __INTERPOLATION = {
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest_exact": cv2.INTER_NEAREST_EXACT,
        "linear_exact": cv2.INTER_LINEAR_EXACT,
    }

    def __init__(
        self,
        size: tuple = (256, 256),
        keep_aspect: bool = False,
        center: bool = False,
        interpolation: str = "area",
        padding_color: int = 0,
    ) -> None:
        super().__init__()
        self._size = size
        self._keep_aspect = keep_aspect
        self._center = center
        self._padding_color = padding_color
        self._interpolation = interpolation

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @profiler
    def run(self, image: np.ndarray) -> np.ndarray:
        """Runs the resizing method.

        Args:
            image (np.ndarray): Currently supports 2-dimensional grayscale images.
        """
        self._validate(image)

        # If keep_aspect is True, we create a square image using padding on the smallest dimension.
        if self._keep_aspect:
            image = self._pad_image(image)

        return cv2.resize(
            image,
            dsize=self._size,
            interpolation=self.__INTERPOLATION[self._interpolation],
        )

    def _validate(self, image: np.ndarray) -> None:
        """Validates image and parameters"""

        # Validate interpolation
        if self._interpolation not in self.__INTERPOLATION.keys():
            msg = f"{self._interpolation} is invalid. Valid values include {self.__INTERPOLATION.keys()}"
            self._logger.exception(msg)
            raise ValueError(msg)

        # Validate image
        if len(image.shape) != 2:
            msg = f"Invalid image shape. Image must be two dimensional. The image has shape {image.shape}"
            self._logger.exception(msg)
            raise ValueError(msg)

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """Pads the image to equal dimensions, i.e., square image"""
        h, w = image.shape
        max_dim = np.max([h, w])

        delta_h = max_dim - h
        delta_w = max_dim - w
        if self._center:
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
        else:
            top, bottom = 0, delta_h
            left, right = 0, delta_w

        image_padded = cv2.copyMakeBorder(
            image,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=self._padding_color,
        )

        return image_padded
