#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/evaluate.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 27th 2023 03:24:36 am                                                #
# Modified   : Monday November 6th 2023 12:45:20 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from math import log10

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssidx

from bcd import DataClass, Stage
from bcd.config import Config
from bcd.image import Image
from bcd.preprocess.image.method.basemethod import Method


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
    mode: str
    stage: Stage
    method: type[Method]
    params: dict
    build_time: float
    mse: float
    psnr: float
    ssim: float
    evaluated: datetime

    def as_dict(self) -> dict:
        return {
            "orig_uid": self.orig.uid,
            "test_uid": self.test.uid,
            "mode": self.mode,
            "stage_id": self.stage.uid,
            "stage": self.stage.name,
            "method": self.method.__name__,
            "params": json.dumps(self.params),
            "build_time": self.build_time,
            "mse": self.mse,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "evaluated": self.evaluated,
        }

    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.as_dict())


# ------------------------------------------------------------------------------------------------ #
#                                    EVALUATOR                                                     #
# ------------------------------------------------------------------------------------------------ #
class Evaluator:
    """Evaluates images"""

    @classmethod
    def evaluate(
        cls,
        orig: Image,
        test: Image,
        stage: Stage,
        method: str,
        params: str,
        mse: type[MSE] = MSE,
        psnr: type[PSNR] = PSNR,
        ssim: type[SSIM] = SSIM,
        config: type[Config] = Config,
    ) -> Evaluation:
        """Creates an evaluation object

        Args:
            orig (Image): Ground truth image object.
            test (Image): The test image object
            stage (Stage): A Stage object encapsulating the stage for which the evaluation
                is being conducted.
            method (str): The name of the method being evaluated.
            params (str): Parameters of the method in string format.
            mse (type[MSE]): The MSE computation class
            psnr (type[PSNR]): The PSNR computation class
            ssim (type[SSIM]): The SSIM computation class
            config (type[Config]): The application configuration class.

        """

        return Evaluation(
            orig=orig,
            test=test,
            mode=config.mode,
            stage=stage,
            method=method,
            params=params,
            build_time=test.build_time,
            mse=mse.compute(orig=orig.pixel_data, test=test.pixel_data),
            psnr=psnr.compute(orig=orig.pixel_data, test=test.pixel_data),
            ssim=ssim.compute(orig=orig.pixel_data, test=test.pixel_data),
            evaluated=datetime.now(),
        )
