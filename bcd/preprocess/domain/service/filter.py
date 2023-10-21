#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/imaging/domain/service/denoise.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 20th 2023 09:31:42 am                                                #
# Modified   : Friday October 20th 2023 10:38:20 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import sys
import logging

import cv2

from ..entity.image import Image

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class Filter(ABC):
    @abstractmethod
    def filter(self, image: Image, *args, **kwargs) -> Image:
        """Executes the filter operation."""


# ------------------------------------------------------------------------------------------------ #
class MeanFilter:
    def filter(self, image: Image, kernel: tuple = (5, 5), *args, **kwargs) -> Image:
        if image.pixel_data is not None:
            image.pixel_data = cv2.blur(image.pixel_data, kernel)
        else:
            msg = "The image has no pixel data"
            logging.exception(msg)
            raise TypeError(msg)
        return image


# ------------------------------------------------------------------------------------------------ #
class MedianFilter:
    def filter(self, image: Image, kernel: int = 3, *args, **kwargs) -> Image:
        if image.pixel_data is not None:
            image.pixel_data = cv2.medianBlur(image.pixel_data, kernel)
        else:
            msg = "The image has no pixel data"
            logging.exception(msg)
            raise TypeError(msg)
        return image


# ------------------------------------------------------------------------------------------------ #
class GaussianFilter:
    def filter(self, image: Image, kernel: tuple = (5, 5), *args, **kwargs) -> Image:
        if image.pixel_data is not None:
            image.pixel_data = cv2.GaussianBlur(image.pixel_data, kernel, 0)
        else:
            msg = "The image has no pixel data"
            logging.exception(msg)
            raise TypeError(msg)
        return image


# ------------------------------------------------------------------------------------------------ #
class BilateralFilter:
    def filter(
        self,
        image: Image,
        diameter: int = 11,
        sigma_color: int = 11,
        sigma_space: int = 11,
        *args,
        **kwargs,
    ) -> Image:
        if image.pixel_data is not None:
            image.pixel_data = cv2.bilateralFilter(
                image.pixel_data, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space
            )
        else:
            msg = "The image has no pixel data"
            logging.exception(msg)
            raise TypeError(msg)
        return image
