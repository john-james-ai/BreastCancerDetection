#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/explore/methods/metrics.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday December 24th 2023 08:54:39 pm                                               #
# Modified   : Thursday January 11th 2024 03:27:10 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from bcd.preprocess.base import Task


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-differ
# ------------------------------------------------------------------------------------------------ #
#                                      MSE                                                         #
# ------------------------------------------------------------------------------------------------ #
class MSE(Task):
    """Computes MSE between two images"""

    def run(self, a: np.ndarray, b: np.ndarray) -> float:
        self.start()
        score = mse(a, b)
        self.stop()
        return score


# ------------------------------------------------------------------------------------------------ #
#                                      PSNR                                                        #
# ------------------------------------------------------------------------------------------------ #
class PSNR(Task):
    """Computes PSNR between two images"""

    def run(self, a: np.ndarray, b: np.ndarray) -> float:
        self.start()
        score = psnr(a, b)
        self.stop()
        return score


# ------------------------------------------------------------------------------------------------ #
#                                      SSIM                                                        #
# ------------------------------------------------------------------------------------------------ #
class SSIM(Task):
    """Computes PSNR between two images"""

    def run(self, a: np.ndarray, b: np.ndarray) -> float:
        self.start()
        score = ssim(a, b)
        self.stop()
        return score
