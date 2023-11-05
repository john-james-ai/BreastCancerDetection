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
# Modified   : Sunday November 5th 2023 12:54:00 am                                                #
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

from bcd import Entity
from bcd.config import Config
from bcd.image import Image
from bcd.preprocess.image.flow.state import Stage
from bcd.utils.date import to_datetime


# ------------------------------------------------------------------------------------------------ #
#                                      METRICS                                                     #
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
#                                    EVALUATION                                                    #
# ------------------------------------------------------------------------------------------------ #
@dataclass(eq=False)
class Evaluation(Entity):
    """Evaluation of Preprocessing Methods"""

    test_no: int
    source_image_uid: str
    source_image_filepath: str
    test_image_uid: str
    test_image_filepath: str
    mode: str
    stage_id: int
    stage: str
    image_view: str
    abnormality_type: str
    assessment: int
    cancer: bool
    method: str
    params: str
    comp_time: float = 0
    mse: float = 0
    psnr: float = 0
    ssim: float = 0
    evaluated: datetime = None

    def __eq__(self, other: Evaluation) -> bool:
        """Evaluates equality of another Evaluation

        Args:
            other (Evaluation): An Evaluation object

        """
        return (
            self.test_no == other.test_no
            and self.source_image_uid == other.source_image_uid
            and self.source_image_filepath == other.source_image_filepath
            and self.test_image_uid == other.test_image_uid
            and self.test_image_filepath == other.test_image_filepath
            and self.mode == other.mode
            and self.stage_id == other.stage_id
            and self.stage == other.stage
            and self.image_view == other.image_view
            and self.abnormality_type == other.abnormality_type
            and self.assessment == other.assessment
            and self.cancer == other.cancer
            and self.method == other.method
            and self.params == other.params
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Evaluation:
        """Creates an evaluation from a DataFrame

        Args:
            df (pd.DataFrame): DataFrame containing the evaluation data.
        """
        evaluated = to_datetime(dt=df["evaluated"].values[0])
        return cls(
            test_no=df["test_no"].values[0],
            source_image_uid=df["source_image_uid"].values[0],
            source_image_filepath=df["source_image_filepath"].values[0],
            test_image_uid=df["test_image_uid"].values[0],
            test_image_filepath=df["test_image_filepath"].values[0],
            mode=df["mode"].values[0],
            stage_id=df["stage_id"].values[0],
            stage=df["stage"].values[0],
            image_view=df["image_view"].values[0],
            abnormality_type=df["abnormality_type"].values[0],
            assessment=df["assessment"].values[0],
            cancer=df["cancer"].values[0],
            method=df["method"].values[0],
            params=df["params"].values[0],
            comp_time=df["comp_time"].values[0],
            mse=df["mse"].values[0],
            psnr=df["psnr"].values[0],
            ssim=df["ssim"].values[0],
            evaluated=evaluated,
        )

    @classmethod
    def evaluate(
        cls,
        test_data: pd.Series,
        orig_image: Image,
        test_image: Image,
        stage_id: int,
        method: str,
        params: str,
        comp_time: float,
        mse: type[MSE] = MSE,
        psnr: type[PSNR] = PSNR,
        ssim: type(SSIM) = SSIM,
    ) -> Evaluation:
        """Creates an evaluation object

        Args:
            test_data (pd.Series): Metadata associated with the image tested.
            orig_image (Image): Ground truth image object.
            test_image (Image): The test image object
            stage_id (int): The stage within the preprocessing cycle
            method (str): The name of the method being evaluated.
            params (str): Parameters of the method in string format.
            comp_time (float): Time taken to execute the transformation.
            mse (type[MSE]): The MSE computation class
            psnr (type[PSNR]): The PSNR computation class
            ssim (type[SSIM]): The SSIM computation class

        """

        return cls(
            test_no=test_data["test_no"],
            source_image_uid=orig_image.uid,
            source_image_filepath=orig_image.filepath,
            test_image_uid=test_image.uid,
            test_image_filepath=test_image.filepath,
            mode=Config.get_mode(),
            stage_id=stage_id,
            stage=Stage(uid=stage_id).name,
            image_view=orig_image.image_view,
            abnormality_type=orig_image.abnormality_type,
            assessment=orig_image.assessment,
            cancer=orig_image.cancer,
            method=method,
            params=params,
            comp_time=comp_time,
            mse=mse.compute(a=orig_image.pixel_data, b=test_image.pixel_data),
            psnr=psnr.compute(a=orig_image.pixel_data, b=test_image.pixel_data),
            ssim=ssim.compute(a=orig_image.pixel_data, b=test_image.pixel_data),
            evaluated=datetime.now(),
        )
