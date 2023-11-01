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
# Modified   : Tuesday October 31st 2023 04:58:39 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from math import log10
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.metrics import structural_similarity as ssidx

from bcd.config import Config
from bcd.core.base import Entity, Method, Param, Stage
from bcd.core.image import Image
from bcd.dal.repo.uow import UoW


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

    image_uid: str
    mode: str
    stage_id: int
    stage: str
    step: str
    method: str
    params: str
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
            and self.params == other.params
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
            params=df["params"].values[0],
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
        params: str,
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
            params=params,
            mse=mse.compute(a=image.pixel_data, b=other),
            psnr=psnr.compute(a=image.pixel_data, b=other),
            ssim=ssim.compute(a=image.pixel_data, b=other),
            image_view=image_view,
            abnormality_type=image.abnormality_type,
            assessment=image.assessment,
            cancer=image.cancer,
            evaluated=datetime.now(),
        )


# ------------------------------------------------------------------------------------------------ #
#                                    EVALUATION                                                    #
# ------------------------------------------------------------------------------------------------ #
class Evaluator:
    """Encapsulates the execution and evaluation of method on a single parameter set."""

    def __init__(self, uow: UoW) -> None:
        self._uow = uow
        self._methods = []
        self._images = []
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def load_images(self, n: int = None, frac: float = None) -> None:
        """Loads a sample of the images from the repository.

        Args:
            n (int): Number of images to select. Can't be used with frac.
            frac (float): Proportion of images to select. Can't be used with n.
        """
        self._uow.connect()
        _, images = self._uow.image_repo.sample(n=n, frac=frac)
        self._uow.close()
        for image in images.values():
            self._images.append(image)

    def add_method(self, method: Method, params: List[Param]) -> None:
        """Adds a method and a parameter list to the Evaluator

        Args:
            method (type[Method]): A Method class.
            params (List[Param]): A list of parameter sets to apply to the method.

        Note: A method can only be added once. A second instance of the method class will
        overwrite the existing method and its parameters.
        """

        d = {"method": method, "params": params}
        self._methods.append(d)

    def run(self) -> None:
        """Runs the method packages on the images, evaluates the transformation and records it."""
        for method_dict in self._methods:
            method = method_dict["method"]
            for params in method_dict["params"]:
                for image in self._images:
                    with Parallel(n_jobs=6) as parallel:
                        parallel(
                            delayed(self._process_image)(method=method, image=image, params=params)
                        )

    def _process_image(self, method: Method, image: Image, params: Param) -> None:
        """Transforms the image, evaluates it, and persists the results."""

        image_transformed = method.execute(image=image.pixel_data, params=params)
        ev = Evaluation.evaluate(
            image=image,
            other=image_transformed,
            stage_id=method.stage_id,
            step=method.step,
            method=method.__name__,
            params=params.to_string(),
        )
        self._logger.debug(ev)
        self._uow.connect()
        self._uow.eval_repo.add(evaluation=ev)
        self._uow.close()
