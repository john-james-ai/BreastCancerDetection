#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/experiment/denoise.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 31st 2023 04:45:05 am                                               #
# Modified   : Wednesday November 29th 2023 11:20:50 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Denoiser Task Module"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from bcd import Stage
from bcd.dal.io.image_reader import ImageReader
from bcd.image import ImageFactory
from bcd.preprocess.image.experiment.base import Experiment
from bcd.preprocess.image.experiment.evaluate import Evaluation
from bcd.preprocess.image.flow.task import Task
from bcd.preprocess.image.method.base import Method
from bcd.utils.image import Noiser


# ------------------------------------------------------------------------------------------------ #
#                                DENOISER EXPERIMENT                                               #
# ------------------------------------------------------------------------------------------------ #
class DenoiseExperiment(Experiment):
    """Denoises images

    Args:
        stage_id (int): Stage of the input data for the experiment
        method (type[Method]): Method class
        params (dict): The parameter grid for the method.
        batchsize (int): Size of batch for reading input.
        reader (ImageReader): ImageReader object.

    """

    def __init__(
        self,
        stage_id: int,
        method: type[Method],
        params: dict,
        batchsize: int = 16,
        var_gaussian: float = 0.01,
        var_speckle: float = 0.01,
        amount: float = 0.01,
        svp: float = 0.5,
        reader: type[ImageReader] = ImageReader,
        noiser: type[Noiser] = Noiser,  # noqa
        evaluation: type[Evaluation] = Evaluation,
        factory: type[ImageFactory] = ImageFactory,
        persist_image: bool = False,
    ) -> None:
        super().__init__()
        self._stage_id = stage_id
        self._method = method
        self._params = params
        self._batchsize = batchsize
        self._reader = reader
        self._noiser = noiser(
            mean=0,
            var_gaussian=var_gaussian,
            var_speckle=var_speckle,
            amount=amount,
            svp=svp,
        )
        self._noise = json.dumps(
            {
                "mean": 0,
                "var_gaussian": var_gaussian,
                "var_speckle": var_speckle,
                "amount": amount,
                "svp": svp,
            }
        )
        self._evaluation = evaluation
        self._factory = factory
        self._persist_image = persist_image

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def run(self) -> None:
        """Runs the Task"""
        # Obtain an iterator for the input stage images
        condition = lambda df: df["stage_id"] == self._stage_id
        reader = self._reader(batchsize=self._batchsize, condition=condition)

        # Obtain a batch from the reader
        for batch in tqdm(reader, desc="batch", total=reader.num_batches):
            # Extract an image from the batch
            for image_in in batch:
                # Add noise to the image
                noisy_pixels = self._noiser.add_noise(image_in.pixel_data)
                # Create param grid
                param_grid = ParameterGrid(self._params)
                # Iterate through the param grid creating tasks
                for param_set in param_grid:
                    task = Task(
                        uid=str(uuid4()),
                        mode=self._config.get_mode(),
                        stage_id=self._stage_id,
                        stage=Stage(uid=self._stage_id).name,
                        method=self._method.__name__,
                        params=param_set,
                        created=datetime.now(),
                    )

                    task.images_processed += 1
                    # Capture the time
                    start = datetime.now()
                    # Execute the method, passing the task.params
                    pixel_data = self._method.execute(noisy_pixels, **task.params)
                    # Compute build time
                    stop = datetime.now()
                    build_time = (stop - start).total_seconds()
                    # Create the output image
                    image_out = self._factory.create(
                        case_id=image_in.case_id,
                        stage_id=self._stage_id,
                        pixel_data=pixel_data,
                        method=self._method.__name__,
                        build_time=build_time,
                        task_id=task.uid,
                    )
                    # Persist the image to the repository
                    if self._persist_image:
                        self._add_image(image=image_out)
                    # Evaluate image quality of image_out vis-a-vis image_in
                    ev = Evaluation.evaluate(
                        orig=image_in,
                        test=image_out,
                        method=self._method.__name__,
                        params=json.dumps(task.params),
                    )
                    # Persist the evaluation.
                    self._add_evaluation(evaluation=ev)
                    # Persist the task
                    self._add_task(task=task)
