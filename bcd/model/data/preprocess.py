#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/layers/preprocess.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday January 9th 2024 01:32:32 pm                                                #
# Modified   : Tuesday January 9th 2024 02:29:27 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Denoiser Module """
import cv2
import numpy as np
from keras import layers


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, useless-parent-delegation, arguments-renamed, arguments-differ
# ------------------------------------------------------------------------------------------------ #
#                             DENOISING WITH MEDIAN FILTER                                         #
# ------------------------------------------------------------------------------------------------ #
class MedianFilter(layers.Layer):
    """Denoises with the median filter

    Args:
        kernel (int): Size of kernel used to compute neighborhood medians.
    """

    def __init__(self, kernel: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._kernel = kernel

    def call(self, x: np.ndarray) -> np.ndarray:
        """Denoises the image using MedianFilter

        Args:
            x (np.ndarray): Image in numpy array format.
        """
        return cv2.medianBlur(x, self._kernel)


# ------------------------------------------------------------------------------------------------ #
#                                   ARTIFACT REMOVER                                               #
# ------------------------------------------------------------------------------------------------ #
class ArtifactRemover(layers.Layers):
    """Removes artifacts from an image"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x: np.ndarray) -> np.narray:
        """Removes artifacts using binary thresholding and contour detection.

        Args:
            x (np.ndarray): Image in numpy array format.
        """
        _, x_bin = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        contours = cv2.findContours(
            x_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        contour_areas = [cv2.contourArea(cont) for cont in contours]
        contour_idx = np.argmax(contour_areas)
        img_mask = cv2.drawContours(
            np.zeros_like(x_bin), contours, contour_idx, 255, -1
        )
        return cv2.bitwise_and(x, x, mask=img_mask)


# ------------------------------------------------------------------------------------------------ #
#                                   CONTRAST ENHANCER                                              #
# ------------------------------------------------------------------------------------------------ #
class ContrastEnhancer(layers.Layer):
    """Uses Contrast Limited Adaptive Histogram Equalization to enhance image contrast"""

    def __init__(
        self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8), **kwargs
    ):
        super().__init__(**kwargs)
        self._clip_limit = clip_limit
        self._tile_grid_size = tile_grid_size

    def call(self, x: np.ndarray) -> np.ndarray:
        """Enhances contrast of image

        Args:
            x (np.ndarray): Image in numpy array format.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit, tileGridSize=self._tile_grid_size
        )
        return clahe.apply(x)
