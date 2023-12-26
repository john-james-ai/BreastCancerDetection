#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/enhance.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday December 26th 2023 03:30:43 am                                              #
# Modified   : Tuesday December 26th 2023 03:48:47 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import cv2
import numpy as np

from bcd import Task


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
#                                      ENHANCE                                                     #
# ------------------------------------------------------------------------------------------------ #
class Enhance(Task):
    """Contrast-Limited Adaptive Histogram Equalization (CLAHE) based Enhancement

    CLAHE enhancement is used to increase the contrast between different structures in order
    to make the pathological and health breast tissue distinct.

    Args:
        clip_limit (float): Threshold for contrast limiting. Default = 0.05 [1]_
        grid_size (tuple): Size of the tiles upon which the histograms equalization
            is conducted [1]_.

    References:

    .. [1] K. Alshamrani, H. A. Alshamrani, F. F. Alqahtani, and B. S. Almutairi,
    “Enhancement of Mammographic Images Using Histogram-Based Techniques for Their Classification
    Using CNN,” Sensors, vol. 23, no. 1, p. 235, Dec. 2022, doi: 10.3390/s23010235.

    """

    def __init__(self, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> None:
        super().__init__()
        self._clip_limit = clip_limit
        self._grid_size = grid_size

    def run(self, image: np.ndarray) -> np.ndarray:
        """Performs the CLAHE operation

        Args:
            image (np.ndarray): 2D grayscale 8-bit image in numpy format.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit, tileGridSize=self._grid_size
        )
        return clahe.apply(image)
