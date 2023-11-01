#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/artifact.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 29th 2023 03:17:29 pm                                                #
# Modified   : Tuesday October 31st 2023 04:58:38 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Removes artifacts from mammogram."""
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from bcd.core.base import Param
from bcd.core.image import Image
from bcd.preprocess.image.base import Transformer


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
#                        ARTIFACT REMOVER - LARGEST CONTOUR                                        #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ArtifactRemoverLargestContourParams(Param):
    """Defines the parameters for the ArtifactRemoverLargestContour Task"""

    image_threshold: int = 128
    otsu_threshold: bool = False
    n_jobs: int = 6


# ------------------------------------------------------------------------------------------------ #
class ArtifactRemoverLargestContour(Transformer):
    """Removes artifacts from mammograms.."""

    name = __qualname__
    module = __name__
    stage_id = 1

    def __init__(
        self,
        task_id: str,
        params: Param,
    ) -> None:
        super().__init__(task_id=task_id)
        self._params = params
        self._images_processed = 0

    @property
    def images_processed(self) -> int:
        return self._images_processed

    def execute(self) -> None:
        """Converts DICOM images to PNG format."""
        images = self.read_images(stage_id=0)

        Parallel(n_jobs=self._params.n_jobs)(
            delayed(self._process_image)(image) for image in tqdm(images, total=len(images))
        )

    def _process_image(self, image: Image) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            image_metadata (pd.DataFrame): DataFrame containing image metadata.
        """
        img_gray = cv2.cvtColor(image.pixel_data, cv2.COLOR_BGR2GRAY)
        img_bin = self._binarize(pixel_data=img_gray)
        img_contour = self._extract_contour(pixel_data=img_bin)
        img_output = self._erase_background()

    def _to_grayscale(self, pixel_data: np.array) -> np.array:
        # Convert to float to avoid overflow or underflow.
        img = img.astype(float)
        # Rescale to gray scale values between 0-255
        img_gray = (img - img.min()) / (img.max() - img.min()) * 255.0
        # Convert to uint
        img_gray = np.uint8(img_gray)
        return img_gray

    def _binarize(self, pixel_data: np.ndarray) -> np.ndarray:
        """Creates a binary image.

        This method supports threshold binarization as well as OTSU threshold
        binarization. Threshold binarization converts all pixel values less than
        or equal to the threshold to 0, all other pixels are converted to one.

        Otsu's Threshold is an automatic image thresholding method that returns
        a single threshold that separates the pixels into foreground and background
        classes.

        """
        if self._params.otsu_threshold:
            _, img_bin = cv2.threshold(pixel_data, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _, img_bin = cv2.threshold(
                pixel_data, thresh=self._params.image_threshold, maxval=255, type=cv2.THRESH_BINARY
            )
        return img_bin

    def _extract_contour(self, img: np.array) -> np.array:
        """Extracts the largest contour"""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        return contour

    def _erase_background(self, img: np.array, contour: np.array) -> np.array:
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        output = cv2.bitwise_and(img, mask)
        return output
