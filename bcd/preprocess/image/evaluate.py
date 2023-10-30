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
# Modified   : Monday October 30th 2023 03:57:59 pm                                                #
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

from bcd.config import Config
from bcd.core.base import Entity, Stage
from bcd.core.image.entity import Image


# ------------------------------------------------------------------------------------------------ #
class Metric(ABC):
    """Base class for metrics for evaluating image quality"""

    @classmethod
    @abstractmethod
    def compute(cls, a: np.ndarray, b: np.ndarray) -> float:
        """Computes and returns the score."""


# ------------------------------------------------------------------------------------------------ #
class MSE(Metric):
    """Mean Squared Error"""

    @classmethod
    def compute(cls, a: np.ndarray, b: np.ndarray) -> None:
        """Computes the score."""
        return np.mean((a.astype("float") - b.astype("float")) ** 2)


# ------------------------------------------------------------------------------------------------ #
class PSNR(Metric):
    """Peak Signal to Noise Ratio"""

    @classmethod
    def compute(cls, a: np.ndarray, b: np.ndarray) -> None:
        """Computes the score."""
        mse = np.mean((a.astype("float") - b.astype("float")) ** 2)
        max_value = np.max(a, axis=None)
        return 20 * log10(max_value / np.sqrt(mse))


# ------------------------------------------------------------------------------------------------ #
class SSIM(Metric):
    """Structural Similarity Index"""

    @classmethod
    def compute(cls, a: np.ndarray, b: np.ndarray) -> None:
        """Computes the score."""
        return ssidx(a, b)


# ------------------------------------------------------------------------------------------------ #
@dataclass(eq=False)
class Evaluation(Entity):
    """Evaluation of Preprocessing Methods"""

    image_uid: str
    mode: str
    stage_id: int
    stage: str
    step: str
    method: str
    mse: float
    psnr: float
    ssim: float
    image_view: str
    abnormality_type: str
    assessment: int
    cancer: bool
    evaluated: datetime = None

    def __eq__(self, other: Evaluation) -> bool:
        """Evaluates equality of another Evaluation

        Args:
            other (Evaluation): An Evaluation object

        """
        return (
            self.image_uid == other.image_uid
            and self.mode == other.mode
            and self.stage_id == other.stage_id
            and self.stage == other.stage
            and self.step == other.step
            and self.method == other.method
            and round(self.mse, 1) == round(other.mse, 1)
            and round(self.psnr, 1) == round(other.psnr, 1)
            and round(self.ssim, 1) == round(other.ssim, 1)
            and self.image_view == other.image_view
            and self.abnormality_type == other.abnormality_type
            and self.assessment == other.assessment
            and self.cancer == other.cancer
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Evaluation:
        """Creates an evaluation from a DataFrame

        Args:
            df (pd.DataFrame): DataFrame containing the evaluation data.
        """
        return cls(
            image_uid=df["image_uid"].values[0],
            mode=df["mode"].values[0],
            stage_id=df["stage_id"].values[0],
            stage=df["stage"].values[0],
            step=df["step"].values[0],
            method=df["method"].values[0],
            mse=df["mse"].values[0],
            psnr=df["psnr"].values[0],
            ssim=df["ssim"].values[0],
            image_view=df["image_view"].values[0],
            abnormality_type=df["abnormality_type"].values[0],
            assessment=df["assessment"].values[0],
            cancer=df["cancer"].values[0],
            evaluated=df["evaluated"].values[0],
        )

    @classmethod
    def evaluate(
        cls,
        image: Image,
        other: np.ndarray,
        stage_id: int,
        step: str,
        method: str,
        mse: type[MSE] = MSE,
        psnr: type[PSNR] = PSNR,
        ssim: type(SSIM) = SSIM,
    ) -> Evaluation:
        """Creates an evaluation object

        Args:
            image (Image): Ground truth image
            other (np.ndarray): The pixel data created by the method
            stage_id (int): The stage within the preprocessing cycle
            step (str): The step within the preprocessing stage
            method (str): The name of the method being evaluated.
        """
        return cls(
            image_uid=image.uid,
            mode=Config.get_mode(),
            stage_id=stage_id,
            stage=Stage(uid=stage_id).name,
            step=step,
            method=method,
            mse=mse.compute(a=image.pixel_data, b=other),
            psnr=psnr.compute(a=image.pixel_data, b=other),
            ssim=ssim.compute(a=image.pixel_data, b=other),
            image_view=image.image_view,
            abnormality_type=image.abnormality_type,
            assessment=image.assessment,
            cancer=image.cancer,
            evaluated=datetime.now(),
        )
