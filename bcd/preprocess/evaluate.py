#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/evaluate.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 27th 2023 03:24:36 am                                                #
# Modified   : Friday October 27th 2023 04:46:38 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from math import log10

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssidx

from bcd.core.uow import UoW
from bcd.infrastructure.io.image import ImageIO
# ------------------------------------------------------------------------------------------------ #
class Metric(ABC):
    """Base class for metrics for evaluating image quality"""

    @abstractmethod
    def score(self) -> float:
        """Returns the computed score"""

    @abstractmethod
    def compute(self, source: np.ndarray, target: np.ndarray) -> None:
        """Computes the score."""


# ------------------------------------------------------------------------------------------------ #
class EvaluatorABC(ABC):
    """Image Quality Evaluator Base Class"""

    @abstractmethod
    def evaluate(self, application: str) -> None:
        """Evaluates a set of images and posts scores to the database.

        Args:
            application (str): An application to evaluate
        """

# ------------------------------------------------------------------------------------------------ #
class MSE(Metric):
    """Mean Squared Error"""
    def __init__(self) -> None:
        super().__init__()
        self._score = None

    def compute(self, source: np.ndarray, target: np.ndarray) -> None:
        """Computes the score."""
        self._score  = np.mean((source.astype("float") - target.astype("float"))**2)
        return self._score

# ------------------------------------------------------------------------------------------------ #
class PSNR(Metric):
    """Peak Signal to Noise Ratio"""
    def __init__(self) -> None:
        super().__init__()
        self._score = None

    def compute(self, source: np.ndarray, target: np.ndarray) -> None:
        """Computes the score."""
        mse  = np.mean((source.astype("float") - target.astype("float"))**2)
        max_value = np.max(source, axis=None)
        self._score = 20 * log10(max_value / np.sqrt(mse))
        return self._score

# ------------------------------------------------------------------------------------------------ #
class SSIM(Metric):
    """Structural Similarity Index """
    def __init__(self) -> None:
        super().__init__()
        self._score = None

    def compute(self, source: np.ndarray, target: np.ndarray) -> None:
        """Computes the score."""
        self._score = ssidx(source, target)


# ------------------------------------------------------------------------------------------------ #
class Evaluator(EvaluatorABC):
    """Image Quality Evaluator"""
    def __init__(self, uow: UoW, mse: MSE, psnr: PSNR, ssim: SSIM, io : ImageIO) -> None:
        super().__init__()
        self._uow = uow
        self._mse = mse
        self._psnr = psnr
        self._ssim = ssim
        self._io = io

    def evaluate(self, application: str) -> None:
        """Driver method that conducts the evaluation."""
        df_scores = pd.DataFrame()

        pairs = self._get_metadata(application=application)
        for _, pair in pairs.iterrows():
            df = self._score(pair)
            df_scores = pd.concat([df_scores,df], axis=0)

        self._uow.score_repo.add(scores=df_scores)

    def _get_metadata(self, application: str) -> pd.DataFrame:
        """Obtains pairwise metadata for the images to be compared."""
        source_vars = ['uid', 'case_id', 'filepath']
        target_vars = ['uid', 'case_id', 'task_id','application', 'filepath']
        source = self._uow.image_repo.get_by_stage(stage_id=0)
        source = source[source_vars]

        target = self._uow.image_repo.get_by_processor(application=application)
        target = target[target_vars]
        return source.merge(target, how='left', on='case_id', suffixes=["_s", "_t"])

    def _score(self, pair: pd.Series) -> None:
        """Scores the image quality metric."""
        source = self._io.read(pair['filepath_s'])
        target = self._io.read(pair['filepath_t'])

        mse = self._mse.compute(source=source, target=target)
        psnr = self._psnr.compute(source=source, target=target)
        ssim = self._ssim.compute(source=source, target=target)

        d_score = {'task_id': pair['task_id'], 'case_id': pair['case_id'],
                   'application': pair['application'],'source_uuid': pair['uuid_s'],
                   'target_uuid': pair['uuid_t'],'mse': mse, 'psnr': psnr, 'ssim': ssim}
        return pd.DataFrame(data=d_score, index=[0])
