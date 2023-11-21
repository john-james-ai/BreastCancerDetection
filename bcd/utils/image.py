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
# Modified   : Saturday November 18th 2023 01:06:31 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Images Utilities"""
import logging
import os

import cv2
import numpy as np
from skimage.util import random_noise


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
class ImageGen:
    """Generates images with random noise of various types."""

    def __init__(self, source_fp: str, dest_dir) -> None:
        self._source_fp = source_fp
        self._source_fn = os.path.basename(source_fp)
        self._dest_dir = dest_dir
        self._source_img = cv2.imread(self._source_fp)
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def gaussian(self, mean: float = 0, var: float = 0.01) -> None:
        """Creates an image with random gaussian noise.

        The image is stored with filename format as follows:
        <dest_dir>/<source_filename>_guassian_mean_<mean>_var_<var>.png


        Args:
            mean (float): Mean of random distribution. Default = 0
            var (float): Variance of random distribution. Default = 0.01
        """
        noise_img = random_noise(self._source_img, mode="gaussian", mean=mean, var=var)
        noise_img = self._convert_uint8(img=noise_img)

        filename = (
            os.path.basename(self._source_fp)
            + "_gaussian_mean_"
            + str(mean)
            + "_var_"
            + str(var)
            + ".png"
        )
        filepath = os.path.join(self._dest_dir, filename)

        cv2.imwrite(filename=filepath, img=noise_img)

        msg = f"Added gaussian noise with mean={mean} and var={var} to {self._source_fn}. Filename is {filename}."
        self._logger.info(msg)

    def poisson(self) -> None:
        """Creates an image with random poisson noise.

        The image is stored with filename format as follows:
        <dest_dir>/<source_filename>_poisson.png
        """
        noise_img = random_noise(self._source_img, mode="poisson")
        noise_img = self._convert_uint8(img=noise_img)

        filename = os.path.basename(self._source_fp) + "_poisson.png"
        filepath = os.path.join(self._dest_dir, filename)

        cv2.imwrite(filename=filepath, img=noise_img)

        msg = f"Added poisson noise to {self._source_fn}. Filename is {filename}."
        self._logger.info(msg)

    def snp(self, amount: float = 0.05, salt_vs_pepper: float = 0.5) -> None:
        """Creates an image with random salt and pepper noise..

        The image is stored with filename format as follows:
        <dest_dir>/<source_filename>_snp_amount_<amount>_svp_<salt_vs_pepper>.png

        Args:
            amount (float): Proportion of image pixels to replace with noise. Default = 0.05
            salt_vs_pepper (float): Proportion of salt vs pepper. Default = 0.5

        """
        noise_img = random_noise(
            self._source_img, mode="s&p", amount=amount, salt_vs_pepper=salt_vs_pepper
        )
        noise_img = self._convert_uint8(img=noise_img)

        filename = (
            os.path.basename(self._source_fp)
            + "_snp_amount_"
            + str(amount)
            + "_svp_"
            + str(salt_vs_pepper)
            + ".png"
        )
        filepath = os.path.join(self._dest_dir, filename)

        cv2.imwrite(filename=filepath, img=noise_img)

        msg = f"Added {str(amount*100)} % Salt and Pepper noise to {self._source_fn}. Filename is {filename}."
        self._logger.info(msg)

    def speckle(self, mean: float = 0, var: float = 0.01) -> None:
        """Creates an image with random speckle noise.

        The image is stored with filename format as follows:
        <dest_dir>/<source_filename>_speckle_mean_<mean>_var_<var>.png


        Args:
            mean (float): Mean of random distribution. Default = 0
            var (float): Variance of random distribution. Default = 0.01
        """
        noise_img = random_noise(self._source_img, mode="speckle", mean=mean, var=var)
        noise_img = self._convert_uint8(img=noise_img)

        filename = (
            os.path.basename(self._source_fp)
            + "_speckle_mean_"
            + str(mean)
            + "_var_"
            + str(var)
            + ".png"
        )
        filepath = os.path.join(self._dest_dir, filename)

        cv2.imwrite(filename=filepath, img=noise_img)

        msg = f"Added speckle noise with mean={mean} and var={var} to {self._source_fn}. Filename is {filename}."
        self._logger.info(msg)

    def _convert_uint8(self, img: np.ndarray) -> np.ndarray:
        """Converts floating point image in [0,1] to uint8 in [0,255]"""
        return np.array(255 * img, dtype="uint8")
