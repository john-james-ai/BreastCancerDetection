#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/prep/image/clean.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 16th 2023 08:47:53 pm                                                #
# Modified   : Tuesday October 17th 2023 07:06:43 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module responsible for Image Cleaning Tasks"""
import numpy as np
import cv2

from bcd.data.prep.image.base import DICOMTask


# ------------------------------------------------------------------------------------------------ #
class DICOMCleaner(DICOMTask):
    """Removes artifacts from a DICOM Image"""

    def __init__(self, img_size: tuple = (256, 256)) -> None:
        super().__init__()
        self._img_size = img_size

    def run(self, img: np.array) -> np.array:
        img_norm = self._normalize(img=img)
        img_bin = self._binarize(img=img_norm)
        img_contour = self._extract_contour(img=img_bin)
        img_foreground = self._erase_background(img=img_norm, contour=img_contour)
        img_crop = self._crop(img=img_foreground, contour=img_contour)
        return self._resize(img=img_crop)

    def _normalize(self, img: np.array) -> np.array:
        if img.max() != 0:
            img = img / img.max()
        img *= 255
        return img.astype(np.uint8)

    def _binarize(self, img: np.array) -> np.array:
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return img_bin

    def _extract_contour(self, img: np.array) -> np.array:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        return contour

    def _erase_background(self, img: np.array, contour: np.array) -> np.array:
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        output = cv2.bitwise_and(img, mask)
        return output

    def _crop(self, img: np.array, contour: np.array):
        x1, x2 = np.min(contour[:, :, 0]), np.max(contour[:, :, 0])
        y1, y2 = np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
        x1, x2 = int(0.99 * x1), int(1.01 * x2)
        y1, y2 = int(0.99 * y1), int(1.01 * y2)
        return img[y1:y2, x1:x2]

    def _resize(self, img: np.array):
        return cv2.resize(img, self._img_size)
