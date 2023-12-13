#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/base.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday December 13th 2023 12:49:11 pm                                            #
# Modified   : Wednesday December 13th 2023 02:18:39 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Preprocessing Base Module"""
from __future__ import annotations
from abc import ABC, abstractmethod

import  numpy as np

from bcd.dal.io.image_reader import ImageReader
# ------------------------------------------------------------------------------------------------ #
class BasePreprocessor(ABC):

    @abstractmethod
    def run(self) -> None:
# ------------------------------------------------------------------------------------------------ #
class Preprocessor(ABC):
    """Preprocessor Base Class"""

    @property
    @abstractmethod
    def preprocessor(self) -> BasePreprocessor:
        """Returns the preprocessor object."""

    @abstractmethod
    def set_image_reader(self, image_reader: ImageReader) -> None:
        """Sets image reader

        Args:
            image_reader (ImageReader): Reads images from the image repository.
        """

    @abstractmethod
    def set_denoiser(self, filter: str = 'median', kernel: int = 3) -> None:
        """Sets the denoiser filter

        Args:
            filter (str):
        """