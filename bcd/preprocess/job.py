#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/job.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday January 11th 2024 08:57:42 am                                              #
# Modified   : Thursday January 11th 2024 03:09:04 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging

import joblib
import pandas as pd
from tqdm import tqdm

from bcd.dal.image import ImageRepo
from bcd.preprocess.base import Job, Task
from bcd.utils.image import grayscale
from bcd.utils.profile import profiler


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-differ
# ------------------------------------------------------------------------------------------------ #
class ImageProcessingJob(Job):
    """Image preprocessing Job"""

    def __init__(self) -> None:
        super().__init__()
        self._repo = None
        self._inputs = None
        self._cpus = None
        self._destination = None
        self._fmt = None
        self._tasks = []
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def add_repo(self, repo: ImageRepo) -> None:
        self._repo = repo

    def add_resources(self, cpus: int = 12) -> None:
        self._cpus = cpus

    def add_inputs(self, inputs: pd.DataFrame) -> None:
        self._inputs = inputs

    def set_outputs(self, destination: str, fmt: str = "png") -> None:
        self._destination = destination
        self._fmt = fmt

    def add_task(self, task: Task) -> None:
        self._tasks.append(task)

    @profiler
    def run(self) -> None:
        self.validate()

        joblib.Parallel(n_jobs=self._cpus)(
            joblib.delayed(self._process_image)(metadata)
            for _, metadata in tqdm(
                self._inputs.iterrows(), total=self._inputs.shape[0]
            )
        )

    def _process_image(self, metadata: pd.Series) -> None:
        """Performs preprocessing on an image."""

        filename = metadata["mmg_id"] + "." + self._fmt

        if self._repo.force or not self._repo.exists(
            destination=self._destination,
            fileset=metadata["fileset"],
            label=metadata["cancer"],
            filename=filename,
        ):
            image = self._repo.get(filepath=metadata["filepath"])

            image = grayscale(image=image)

            for task in self._tasks:
                image = task.run(image=image)

            self._repo.add(
                image=image,
                destination=self._destination,
                fileset=metadata["fileset"],
                label=metadata["cancer"],
                filename=filename,
            )

    def validate(self) -> None:
        if self._repo is None:
            msg = "The image repository has not been added."
            self._logger.exception(msg)
            raise ValueError(msg)

        if self._inputs is None:
            msg = "Inputs have not been added."
            self._logger.exception(msg)
            raise ValueError(msg)

        if self._cpus is None:
            msg = "CPU resources not set."
            self._logger.exception(msg)
            raise ValueError(msg)

        if len(self._tasks) == 0:
            msg = "No tasks have been added."
            self._logger.exception(msg)
            raise ValueError(msg)

        if self._destination is None:
            msg = "Output destination has not been set."
            self._logger.exception(msg)
            raise ValueError(msg)
