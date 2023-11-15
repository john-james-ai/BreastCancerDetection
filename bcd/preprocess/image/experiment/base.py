#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/experiment/base.py                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 11:03:59 pm                                             #
# Modified   : Monday November 13th 2023 10:28:58 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the Interface for Task classes."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4

from dependency_injector.wiring import Provide, inject
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from bcd import Stage
from bcd.config import Config
from bcd.container import BCDContainer
from bcd.dal.io.image_reader import ImageReader
from bcd.dal.repo.uow import UoW
from bcd.image import ImageFactory
from bcd.preprocess.image.experiment.evaluate import Evaluation
from bcd.preprocess.image.flow.task import Task
from bcd.preprocess.image.method.base import Method


# ------------------------------------------------------------------------------------------------ #
class Experiment(ABC):
    """Abstract base class for task objects.

    Args:
        instage_id (int): Stage of the input data for the experiment
        outstage_id (int): Stage of the output data for the experiment
        method (type[Method]): Method class
        params (dict): The parameter grid for the method.
        batchsize (int): Size of batch for reading input.
        uow (UoW): Unit of work class containing repositories.
        config (type[Config]): App config class
        reader (type[ImageReader]): Imagereader class
        factory (type[ImageFactory]): Image Factory class

    """

    @inject
    def __init__(
        self,
        instage_id: int,
        outstage_id: int,
        method: type[Method],
        params: dict,
        batchsize: int = 16,
        uow: UoW = Provide[BCDContainer.dal.uow],
        config: type[Config] = Config,
        reader: type[ImageReader] = ImageReader,
        factory: type[ImageFactory] = ImageFactory,
    ) -> None:
        self._instage_id = instage_id
        self._outstage_id = outstage_id
        self._method = method
        self._params = params
        self._batchsize = batchsize
        self._uow = uow
        self._config = config
        self._reader = reader
        self._factory = factory()

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def run(self) -> None:
        """Runs the Task"""
        # Obtain an iterator for the input stage images
        condition = lambda df: df["stage_id"] == self._instage_id
        reader = self._reader(batchsize=self._batchsize, condition=condition)

        # Construct a task list, one task for each parameter set
        tasks = self._get_tasks()

        # Obtain a batch from the reader
        for batch in tqdm(reader, desc="batch", total=reader.num_batches):
            # Extract an image from the batch
            for image_in in tqdm(batch, desc="image", total=len(batch)):
                # Iterate through each task, which contains method parameters
                for task in tqdm(tasks, desc="task", total=len(tasks)):
                    task.images_processed += 1
                    # Capture the time
                    start = datetime.now()
                    # Execute the method, passing the task.params
                    pixel_data = self._method.execute(
                        image_in.pixel_data, **task.params
                    )
                    # Compute build time
                    stop = datetime.now()
                    build_time = (stop - start).total_seconds()
                    # Create the output image
                    image_out = self._factory.create(
                        case_id=image_in.case_id,
                        stage_id=self._outstage_id,
                        pixel_data=pixel_data,
                        method=self._method.__name__,
                        build_time=build_time,
                        task_id=task.uid,
                    )
                    # Persist the image to the repository
                    self._uow.image_repo.add(image=image_out)
                    # Evaluate image quality of image_out vis-a-vis image_in
                    ev = Evaluation.evaluate(
                        orig=image_in,
                        test=image_out,
                        method=self._method.__name__,
                        params=json.dumps(self._params),
                    )
                    # Persist the evaluation.
                    self._uow.eval_repo.add(evaluation=ev)
                    # Persist the task
                    self._uow.task_repo.add(task=task)

    def _get_tasks(self) -> list:
        """Returns a list of parameter sets based upon the param(grid)"""
        task_list = []
        param_grid = ParameterGrid(self._params)
        for param_set in param_grid:
            task = Task(
                uid=str(uuid4()),
                mode=self._config.get_mode(),
                stage_id=self._outstage_id,
                stage=Stage(uid=self._outstage_id).name,
                method=self._method.__name__,
                params=param_set,
                created=datetime.now(),
            )
            task_list.append(task)
        return task_list
