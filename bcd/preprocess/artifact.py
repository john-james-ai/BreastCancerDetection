#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/artifact.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday December 25th 2023 09:13:38 pm                                               #
# Modified   : Thursday January 11th 2024 03:27:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Artifact removal module"""
import cv2
import numpy as np

from bcd.preprocess.base import Task
from bcd.preprocess.threshold import Threshold
from bcd.utils.image import grayscale


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
#                             ARTIFACT REMOVER CONTOUR-BASED                                       #
# ------------------------------------------------------------------------------------------------ #
class ArtifactRemoverContour(Task):
    """Removes artifacts from grayscale images using a border following algorithm.

    This algorithm segments the breast from the background and artifacts by first, binarizing the
    image using a global threshold. Then, the contours are extracted using a border-following
    algorithm. Areas of each contour are computed, and the contour with the largest area is selected
    as that which contains the breast tissue. A contour drawing algorithm connects and smooths the
    contour and produces a binary mask. This mask is applied to the original image.

    Args:
        binarizer (Threshold): A Threshold object.

    Returns:
        np.ndarray: Image with artifacts removed.

    References:
    .. [1] S. Suzuki and K. Be, “Topological structural analysis of digitized
       binary images by border following,” Computer Vision, Graphics, and Image
       Processing, vol. 30, no. 1, pp. 32–46, Apr. 1985,
       doi: 10.1016/0734-189X(85)90016-7.


    """

    def __init__(self, binarizer: Threshold) -> None:
        super().__init__()
        self._binarizer = binarizer

    def run(self, image: np.ndarray) -> np.ndarray:
        """Remove artifacts

        Args:
            image (np.ndarray): Image in numpy array format.
        """
        # Ensure image is grayscale in 2 dimensions only.
        # DICOM images have three dimensions.
        image = grayscale(image)

        _, image_bin = self._binarizer.run(image=image)
        # Extract contours using border following algorithm
        contours, _ = cv2.findContours(
            image=image_bin.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
        # Compute areas for the contours, and obtain index
        # for largest contour
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        idx = np.argmax(contour_areas)

        # Create a breast mask and apply it to the original image
        mask = cv2.drawContours(
            np.zeros_like(image_bin),
            contours=contours,
            contourIdx=idx,
            color=255,
            thickness=-1,
        )
        image_seg = cv2.bitwise_and(image, image, mask=mask)

        return image_seg


# ------------------------------------------------------------------------------------------------ #
#                             ARTIFACT REMOVER CONNECTION-BASED                                    #
# ------------------------------------------------------------------------------------------------ #
class ArtifactRemoverConnection(Task):
    """Removes artifacts from grayscale images

    This algorithm segments the breast from the background and artifacts using
    connected components labeling. Connected components are labeled with statistics. The
    largest object is selected based upon its area statistic. Finally, a binary mask is
    created.

    Args:
        binarizer (Threshold): A Threshold object.

    Returns:
        np.ndarray: Image with artifacts removed.

    """

    def __init__(
        self,
        binarizer: Threshold,
        crop: bool = False,
        fill_holes: bool = False,
        fill_color: int = 255,
        smooth_boundary: bool = False,
        kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self._binarizer = binarizer
        self._crop = crop
        self._fill_holes = fill_holes
        self._fill_color = fill_color
        self._smooth_boundary = smooth_boundary
        self._kernel_size = kernel_size

    def run(self, image: np.ndarray) -> np.ndarray:
        """Remove artifacts

        Args:
            image (np.ndarray): Image in numpy array format.
        """
        # Ensure image is grayscale in 2 dimensions only.
        # DICOM images have three dimensions.
        image = grayscale(image)

        _, image_bin = self._binarizer.run(image)
        output = cv2.connectedComponentsWithStats(
            image_bin, connectivity=8, ltype=cv2.CV_32S
        )
        (_, labels, stats, _) = output
        # Get the index for which the area is largest. We start at 1
        # because the first index is background which we skip. We have
        # to add 1 back to the index to get that which is largest
        # but not background.
        largest_object_label = np.argmax(stats[1:, 4]) + 1

        # Initialize a mask, then set the labeled pixels
        # to the fill color.
        largest_mask = np.zeros(image_bin.shape, dtype=np.uint8)
        largest_mask[labels == largest_object_label] = self._fill_color

        if self._fill_holes:
            largest_mask = self._fill_mask_holes(labels=labels, mask=largest_mask)

        if self._smooth_boundary:
            largest_mask = self._smooth_mask_boundary(mask=largest_mask)

        image_seg = cv2.bitwise_and(image, largest_mask)

        return image_seg

    def _fill_mask_holes(self, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fills holes in the mask"""
        background_idx = np.where(labels == 0)
        # Obtain a seed point in the background.
        background_seed = (background_idx[0][0], background_idx[1][0])

        # Create a copy of the mask that will be passed
        # to floodfill.
        image_ff = mask.copy()

        h, w = mask.shape
        # We add 2 to the height and width because floodfill checks each
        # pixels neighbors and for connection and adding zeros to the
        # height and width, avoids having special cases for border pixels.
        mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(
            image_ff, mask_ff, seedPoint=background_seed, newVal=self._fill_color
        )
        mask_holes = cv2.bitwise_not(image_ff)
        # Fill the holes
        mask = mask + mask_holes
        return mask

    def _smooth_mask_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Smooths mask boundary"""

        kernel = np.ones((self._kernel_size, self._kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel)
