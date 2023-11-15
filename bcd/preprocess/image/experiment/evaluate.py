#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/experiment/evaluate.py                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 27th 2023 03:24:36 am                                                #
# Modified   : Monday November 13th 2023 02:18:36 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from math import log10

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssidx

from bcd import DataClass
from bcd.image import Image


# ------------------------------------------------------------------------------------------------ #
#                                      METRICS                                                     #
# ------------------------------------------------------------------------------------------------ #
class Metric(ABC):
    """Base class for metrics for evaluating image quality"""

    @classmethod
    @abstractmethod
    def compute(cls, orig: np.ndarray, test: np.ndarray) -> float:
        """Computes and returns the score."""


# ------------------------------------------------------------------------------------------------ #
class MSE(Metric):
    """Mean Squared Error"""

    @classmethod
    def compute(cls, orig: np.ndarray, test: np.ndarray) -> float:
        """Computes the score."""
        return np.mean((orig.astype("float") - test.astype("float")) ** 2)


# ------------------------------------------------------------------------------------------------ #
class PSNR(Metric):
    """Peak Signal to Noise Ratio"""

    @classmethod
    def compute(cls, orig: np.ndarray, test: np.ndarray) -> float:
        """Computes the score."""
        mse = np.mean((orig.astype("float") - test.astype("float")) ** 2)
        max_value = np.max(orig, axis=None)
        return 20 * log10(max_value / np.sqrt(mse))


# ------------------------------------------------------------------------------------------------ #
class SSIM(Metric):
    """Structural Similarity Index"""

    @classmethod
    def compute(cls, orig: np.ndarray, test: np.ndarray) -> float:
        """Computes the score."""
        return ssidx(orig, test)


# ------------------------------------------------------------------------------------------------ #
#                                   EVALUATION                                                     #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Evaluation(DataClass):
    """Encapsulates an image evaluation."""

    orig: Image
    test: Image
    method: str
    params: str
    build_time: float
    mse: float
    psnr: float
    ssim: float
    evaluated: datetime

    def as_dict(self) -> dict:
        return {
            "orig_uid": self.orig.uid,
            "test_uid": self.test.uid,
            "mode": self.test.mode,
            "stage_id": self.test.stage_id,
            "stage": self.test.stage,
            "method": self.method,
            "params": self.params,
            "image_view": self.test.image_view,
            "abnormality_type": self.test.abnormality_type,
            "assessment": self.test.assessment,
            "cancer": self.test.cancer,
            "build_time": self.build_time,
            "task_id": self.test.task_id,
            "mse": self.mse,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "evaluated": self.evaluated,
        }

    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.as_dict(), index=[0])

    @classmethod
    def evaluate(
        cls,
        orig: Image,
        test: Image,
        method: str,
        params: str,
        mse: type[MSE] = MSE,
        psnr: type[PSNR] = PSNR,
        ssim: type[SSIM] = SSIM,
    ) -> Evaluation:
        """Creates an evaluation object

        Args:
            orig (Image): Ground truth image object.
            test (Image): The test image object
            method (str): The name of the method being evaluated.
            params (str): The parameters used by the method.
            mse (type[MSE]): The MSE computation class
            psnr (type[PSNR]): The PSNR computation class
            ssim (type[SSIM]): The SSIM computation class
            config (type[Config]): The application configuration class.

        """

        return cls(
            orig=orig,
            test=test,
            method=method,
            params=params,
            build_time=test.build_time,
            mse=mse.compute(orig=orig.pixel_data, test=test.pixel_data),
            psnr=psnr.compute(orig=orig.pixel_data, test=test.pixel_data),
            ssim=ssim.compute(orig=orig.pixel_data, test=test.pixel_data),
            evaluated=datetime.now(),
        )
