#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/builder.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday January 11th 2024 08:42:26 am                                              #
# Modified   : Thursday January 11th 2024 03:59:36 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
from typing import Callable

from bcd.dal.image import ImageRepo
from bcd.preprocess.base import Job, JobBuilder, Task
from bcd.preprocess.job import ImageProcessingJob


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-differ
# ------------------------------------------------------------------------------------------------ #
class ImageProcessingJobBuilder(JobBuilder):
    """Job builder for image preprocessing

    Args:
        repo (type[ImageRepo]): ImageRepo class
        cpus (int): Default number of cpus for parallelism. Default = 12
    """

    __image_cols = ["fileset", "filepath", "cancer", "mmg_id"]

    def __init__(self, repo: type[ImageRepo] = ImageRepo, cpus: int = 12) -> None:
        super().__init__()
        self._repo = repo
        self._cpus = cpus
        self._job = None

        self._n = None
        self._frac = None
        self._condition = None
        self._groupby = None
        self._random_state = None
        self._destination = None
        self._force = None
        self._fmt = None
        self._tasks = []
        self.reset()

    def reset(self) -> None:
        self._job = ImageProcessingJob()
        self._n = None
        self._frac = None
        self._condition = None
        self._groupby = None
        self._random_state = None
        self._destination = None
        self._force = None
        self._fmt = None
        self._tasks = []

    @property
    def job(self) -> Job:
        self._finalize_job()
        job = self._job
        self.reset()
        return job

    def set_resources(self, cpus: int = 12) -> None:
        """Sets the number of cpus for parallelism

        Args:
            cpus (int): Number of cpus for parallelism. Default = 12
        """
        self._cpus = cpus

    def set_inputs(
        self,
        n: int = None,
        frac: float = None,
        condition: Callable = None,
        groupby: list = None,
        random_state: int = None,
    ) -> None:
        """Sets the input for the job.

        Args:
            n (int): Number of records to return or the number of records
                to return in each group, if groupby is not None. Default = None
            frac (float): The fraction of all records to return or the
                fraction of records by group if groupby is not None. Ignored
                if n is not None. Default = None
            condition (Callable): Lambda expression used to subset the metadata.
                Default is None
            groupby (list): List of grouping variables for stratified sampling.
                Default is None.

            If all parameters are None, all records are returned.
        """
        self._n = n
        self._frac = frac
        self._condition = condition
        self._groupby = groupby
        self._random_state = random_state

    def set_outputs(
        self, destination: str, fmt: str = "png", force: bool = False
    ) -> None:
        self._destination = destination
        self._fmt = fmt
        self._force = force

    def add_task(self, task: Task) -> None:
        """Adds a task to the Job builder."""
        self._tasks.append(task)

    def _finalize_job(self) -> None:
        """Finalizes the Job object."""
        # Obtain the metadata for selected images from the repository.
        repo = self._repo(force=self._force)
        inputs = repo.query(
            n=self._n,
            frac=self._frac,
            condition=self._condition,
            groupby=self._groupby,
            random_state=self._random_state,
            columns=self.__image_cols,
        )

        # Build the job.
        self._job.add_repo(repo=repo)
        self._job.add_resources(cpus=self._cpus)
        self._job.add_inputs(inputs=inputs)
        self._job.set_outputs(destination=self._destination, fmt=self._fmt)
        for task in self._tasks:
            self._job.add_task(task=task)
        self._job.validate()
