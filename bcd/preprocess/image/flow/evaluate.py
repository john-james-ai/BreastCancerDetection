#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/evaluate.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday November 1st 2023 07:40:20 pm                                             #
# Modified   : Wednesday November 1st 2023 09:02:13 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Evaluator Task Module"""
import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from bcd.config import Config
from bcd.dal.io.image import ImageIO
from bcd.preprocess.image.evaluate import Evaluation
from bcd.preprocess.image.flow.basetask import Task
from bcd.preprocess.image.flow.decorator import counter, timer
from bcd.preprocess.image.image import Image
from bcd.preprocess.image.method.basemethod import Param

# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
#                                   CONVERTER TASK PARAMS                                          #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class EvaluatorTaskParams(Param):
    """Encapsulates the parameters for the ConverterTask."""

    input_stage_id: int
    n: int = None
    frac: float = None
    groupby: Union[str, list] = None
    n_jobs: int = 6
    random_state: int = None

    def __post_init__(self) -> None:
        if (self.n is not None and self.n != "NoneType") and (
            self.frac is not None and self.frac != "NoneType"
        ):
            msg = "Cannot provide both n and frac. One, the other or both must be None"
            logging.exception(msg)
            raise ValueError(msg)


# ------------------------------------------------------------------------------------------------ #
#                                   EVALUATOR TASK                                                 #
# ------------------------------------------------------------------------------------------------ #
class EvaluatorTask(Task):
    """Evaluates a preprocessing method by comparing output image quality with ground truth image.

    Args:
        task_params (ConverterTaskParams): Parameters that control the task behavior.
        config (Config): The application configuration class
        io (ImageIO): The class responsible for image io
    """

    def __init__(
        self,
        config: Config = Config,
        io: ImageIO = ImageIO,
    ) -> None:
        super().__init__()
        self._config = config
        self._io = io

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @timer
    def run(self) -> None:
        self.start()
        source_image_metadata = self._get_source_image_metadata()
        self._process_images(image_metadata=source_image_metadata)
        self.stop()

    def _get_source_image_metadata(self) -> pd.DataFrame:
        """Performs multivariate stratified sampling to obtain a fraction of the raw images."""

        # Read the metadata from image repository
        source_image_metadata, _ = self.uow.image_repo.get_by_stage(
            stage_id=self.task_params.input_stage_id
        )

        # Execute the sampling and obtain the case_ids
        if self.task_params.groupby is None:
            df = source_image_metadata.sample(
                n=self.task_params.n,
                frac=self.task_params.frac,
                random_state=self.task_params.random_state,
            )
        else:
            df = source_image_metadata.groupby(by=self.task_params.groupby).sample(
                n=self.task_params.n,
                frac=self.task_params.frac,
                random_state=self.task_params.random_state,
            )

        return df

    def _process_images(self, image_metadata: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            image_metadata (pd.DataFrame): DataFrame containing image metadata.
        """
        for _, metadata in tqdm(image_metadata.iterrows(), total=image_metadata.shape[0]):
            self._process_image(metadata=metadata)

    @counter
    def _process_image(self, metadata: pd.Series) -> None:
        # Get the image from the repository
        image = self.uow.image_repo.get(uid=metadata["uid"])
        # Execute the method being evaluated
        pixel_data = self._method.execute(image=image.pixel_data, params=self.method_params)
        # Perform the evaluation
        ev = self._evaluate(image=image, pixel_data=pixel_data)
        # Persist
        self.uow.eval_repo.add(evaluation=ev)

    def _evaluate(self, image: Image, pixel_data: np.ndarray) -> Evaluation:
        """Obtains the ground truth image if necessary and performs the evaluation."""
        ground_truth = self._get_ground_truth(image=image)
        ev = Evaluation.evaluate(
            image=ground_truth,
            other=pixel_data,
            method=self.method.__class__.__name__,
            stage_id=self.method.stage.uid,
            step=self.method.step,
            params=self.method_params.as_string(),
        )
        return ev

    def _get_ground_truth(self, image: Image) -> Image:
        """Returns the ground truth image, if necessary"""
        if image.stage_id == 0:
            return image
        else:
            metadata = self.uow.image_repo.get_by_stage(stage_id=0)
            image_meta = metadata.loc[metadata["case_id"] == image.case_id]
            if len(image_meta) == 0:
                msg = f"There is no ground truth image for case_id: {image.case_id}."
                self._logger.exception(msg)
                raise FileNotFoundError(msg)
            elif len(image_meta) > 1:
                msg = f"There is more than one ground truth image for case_id: {image.case_id}."
                self._logger.exception(msg)
                raise RuntimeError(msg)
            else:
                return self.uow.image_repo.get(image_meta["uid"].values[0])
