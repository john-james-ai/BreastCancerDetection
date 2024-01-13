#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/pectoral.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday December 24th 2023 11:04:18 pm                                               #
# Modified   : Thursday January 11th 2024 03:27:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module for Pectoral Removal"""
import cv2
import numpy as np
from skimage.draw import polygon

from bcd.preprocess.base import Task
from bcd.utils.image import orient_image


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
#                                  PECTORAL MUSCLE REMOVER                                         #
# ------------------------------------------------------------------------------------------------ #
class PectoralRemover(Task):
    """Removes Pectoral Muscle from Mammogram.

    Canny Edge Detection is used to detect the edges within a mammogram. Hough lines are
    extracted and those lines which approximate the spatial characteristics of the
    border of the pectoral muscle are shortlisted and ranked. The top listed line is used
    to create a binary mask containing the pectoral removal. A bitwise and operation suppresses
    the pectoral removal from the original image.

    Args:
        canny_sigma (float): Value that determines the threshold width for automated
            Canny Edge Detection. Default = 0.1
        houghlines_max_threshold (int): The maximum number of hough line transform intersections
            that are required for a series of points to be designated a line. The process
            of selecting lines starts with this number and it is iteratively decreased until
            we have one or more lines that meet the shortlist criteria. Default = 150
        min_houghline_candidates (int): Number of minimum candidate hough lines. Default = 5

    References:
    .. [1] A. R. Beeravolu, S. Azam, M. Jonkman, B. Shanmugam, K. Kannoorpatti, and A. Anwar,
          “Preprocessing of Breast Cancer Images to Create Datasets for Deep-CNN,”
          IEEE Access, vol. 9, pp. 33438–33463, 2021, doi: 10.1109/ACCESS.2021.3058773.

    """

    # Default screening characteristics for pectoral border line candidates.
    __MIN_ANGLE = 10
    __MAX_ANGLE = 70
    __MIN_LENGTH = 5
    __MAX_LENGTH = 256

    def __init__(
        self,
        canny_sigma: float = 0.1,
        houghlines_max_threshold: int = 150,
        min_houghline_candidates: int = 1,
    ):
        super().__init__()
        self._canny_sigma = canny_sigma
        self._houghlines_max_threshold = houghlines_max_threshold
        self._min_houghline_candidates = min_houghline_candidates
        self._image_shape = None

    def run(self, image: np.ndarray) -> np.ndarray:
        """Performs the pectoral removal operation

        Args:
            image (np.ndarray): 2-d 8-bit grayscale square, preferably a contrast enhanced image
                in numpy format.
        """
        self._image_shape = image.shape
        # Ensure the image is a 2-dimensional grayscale image with aspect ratio of 1:1.
        self._validate(image=image)
        # Orient the MLO image so that the breast ROI is left oriented.
        image, flipped = orient_image(image=image, left=True)

        edges = self._detect_edges(image=image, auto=True, sigma=self._canny_sigma)

        candidates = self._get_hough_lines(image=edges)

        image_seg = self._remove_pectoral(image=image, candidates=candidates)
        # If the image was flipped, flip it back to the original orientation.
        if flipped:
            image_seg = cv2.flip(image_seg, 1)
        return image_seg

    def _detect_edges(
        self,
        image: np.ndarray,
        auto: bool = True,
        sigma: float = 0.33,
        lower: int = 50,
        upper: int = 200,
    ):
        """Performs manual or automated Canny Edge Detection.

        For manual Canny Edge Detection, a lower and upper threshold must be provided. For automated
        Canny Edge Detection, only sigma needs to be set. The thresholds are based upon
        the median value of the pixel intensities. These thresholds are constructed
        based on +/- percentages controlled by the sigma argument.

        Args:
            image (np.ndarray): One-channel grayscale image in numpy format.
            auto (bool): Whether to use automated Canny Edge Detection. Default is True
            sigma (float): Controls the range between lower and upper threshold. Default = 0.33 [1]_
            lower (int): Lower threshold for the manual setting.
            upper (int): Upper threshold for the manual setting.

        Returns:
            np.ndarray: Canny edges

        Reference: .. [1] A. Rosebrock, “Zero-parameter, automatic Canny edge detection with Python
                      and OpenCV,” PyImageSearch. Accessed: Dec. 26, 2023. [Online]. Available:
        https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/


        """
        if auto:
            # compute the median of the single channel pixel intensities
            v = np.median(image)
            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(np.max(image), (1.0 + sigma) * v))
            edged = cv2.Canny(
                image, threshold1=lower, threshold2=upper, L2gradient=True
            )
        else:
            edged = cv2.Canny(
                image, threshold1=lower, threshold2=upper, L2gradient=True
            )
        # return the edged image
        return edged

    def _get_hough_lines(self, image: np.ndarray) -> list:
        """From edge detected image, extract lines from hough lines transform

        Args:
            image (np.ndarray): Output image from Canny Edge Detected in numpy array
                format.

        Returns:
            list: List of candidate lines matching characteristics of pectoral muscle border.
        """
        threshold = self._houghlines_max_threshold
        # List of pectoral muscle border line candidates
        candidates = []

        # We start with a high threshold, then iteratively lower the threshold
        # until we have lines in the shortlist.
        while len(candidates) < self._min_houghline_candidates:
            hlines = cv2.HoughLines(
                image=image, rho=1, theta=np.pi / 180, threshold=threshold
            )
            if hlines is not None:
                for hline in hlines:
                    line = self._describe_hline(hline=hline)
                    if self._is_candidate(line):
                        candidates.append(line)
            # Drop the threshold by 5% on the next iteration if no candidates
            # are selected.
            threshold = int(threshold * 0.95)
        self.logger.debug(msg=f"Candidate lines\n{candidates}")
        return candidates

    def _describe_hline(self, hline: list) -> dict:
        """Describes a hough line in terms of polar and cartesian coordinates.

        Reference:
        .. [1] “OpenCV: Hough Line Transform.” Accessed: Dec. 26, 2023. [Online].
        Available: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html

        """
        rho, theta = hline[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))
        return {
            "rho": rho,
            "theta": np.degrees(theta),
            "point1": [x1, y1],
            "point2": [x2, y2],
        }

    def _is_candidate(self, line: dict) -> bool:
        """Determines if line matches spatial characteristics of a pectoral muscle border."""
        max_length = max(self.__MAX_LENGTH, self._image_shape[0])
        return (
            (line["rho"] >= self.__MIN_LENGTH)
            & (line["rho"] <= max_length)
            & (line["theta"] >= self.__MIN_ANGLE)
            & (line["theta"] <= self.__MAX_ANGLE)
        )

    def _remove_pectoral(self, image: np.ndarray, candidates: list) -> np.ndarray:
        """Removes pectoral muscle from the image."""
        # Sorts the candidates in place by line length and select the candidate with
        # the greatest length.
        candidates.sort(key=lambda x: x["rho"])
        pectoral_line = candidates[0]

        # Create the polygon which represents the pectoral muscle.
        length = pectoral_line["rho"]
        theta = np.radians(pectoral_line["theta"])

        x_intercept = min(length / np.cos(theta), 255)
        y_intercept = min(length / np.sin(theta), 255)

        rr, cc = polygon([0, 0, y_intercept], [0, x_intercept, 0])
        # Mask the pectoral region
        image[rr, cc] = 0
        return image

    def _validate(self, image: np.ndarray) -> None:
        """Validates the image"""
        if len(image.shape) != 2:
            msg = f"Image must be two dimensional. Image has {len(image.shape)} dimensions."
            self.logger.exception(msg)
            raise TypeError(msg)

        if image.dtype != np.uint8:
            msg = (
                f"Image dtype {image.dtype} is invalid. Images must be uint8 data type."
            )
            self.logger.exception(msg)
            raise TypeError(msg)

        if image.shape[0] != image.shape[1]:
            msg = "Images must have an aspect ratio of 1:1, i.e. square image."
            self.logger.exception(msg)
            raise ValueError(msg)
