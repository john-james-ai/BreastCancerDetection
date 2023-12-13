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
# Modified   : Wednesday December 13th 2023 01:44:29 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Denoiser Task Module"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
import skimage
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from bcd.dal.io.image_reader import ImageReader
from bcd.image import ImageFactory
from bcd.preprocess.image.experiment.base import Experiment
from bcd.preprocess.image.experiment.evaluate import Evaluation
from bcd.preprocess.image.operation.base import Operation
from bcd.utils.image import convert_uint8


# ------------------------------------------------------------------------------------------------ #
#                                DENOISER EXPERIMENT                                               #
# ------------------------------------------------------------------------------------------------ #
class DenoiseExperiment(Experiment):
    """Denoises images

    Args:
        stage_in (int): Stage identifier for the input data for the experiment
        stage_out (int): Stage identifier for the output image

        batchsize (int): Size of batch for reading input.
        reader (ImageReader): ImageReader object.

    """

    def __init__(
        self,
        stage_in: int,
        stage_out: int,
        batchsize: int = 16,
        reader: type[ImageReader] = ImageReader,
        evaluation: type[Evaluation] = Evaluation,
        factory: type[ImageFactory] = ImageFactory,
        persist_image: bool = False,
    ) -> None:
        super().__init__()
        self._stage_in = stage_in
        self._stage_out = stage_out
        self._batchsize = batchsize
        self._reader = reader
        self._evaluation = evaluation
        self._factory = factory
        self._persist_image = persist_image
        self._random_noisers = []

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def add_random_noise_gen(self, random_noise_gen: RandomNoise) -> None:
        self._random_noisers.append(random_noise_gen)

    def run(self, method: type[Operation], params: dict) -> None:
        """Runs the Experiment

        Args:
            method (type[Operation]): Method class
            params (dict): The parameter grid for the method.
        """
        # Confirm that random noisers have been configured.
        if len(self._random_noisers) == 0:
            msg = "Random noise generators have not been provisioned."
            self._logger.exception(msg)
            raise RuntimeError(msg)

        # Obtain an iterator for the input stage images
        condition = lambda df: df["stage_id"] == self._stage_in
        reader = self._reader(batchsize=self._batchsize, condition=condition)

        # Obtain a batch from the reader
        for batch in tqdm(reader, desc="batch", total=reader.num_batches):
            # Extract an image from the batch
            for image_in in batch:
                # Add random noise
                for random_noise in self._random_noisers:
                    noisy_pixels = random_noise.add_noise(image=image_in.pixel_data)

                    # Create param grid
                    param_grid = ParameterGrid(params)
                    # Iterate through the param grid creating tasks
                    for param_set in param_grid:
                        # Capture the time
                        start = datetime.now()
                        # Execute the denoising method, passing the params
                        pixel_data = method.execute(noisy_pixels, **param_set)
                        # Compute build time
                        stop = datetime.now()
                        build_time = (stop - start).total_seconds()
                        # Create the output image
                        image_out = self._factory.create(
                            case_id=image_in.case_id,
                            stage_id=self._stage_out,
                            pixel_data=pixel_data,
                            method=method.__name__,
                            build_time=build_time,
                        )
                        # Persist the image to the repository
                        if self._persist_image:
                            self._add_image(image=image_out)
                        # Evaluate image quality of image_out vis-a-vis image_in
                        ev = Evaluation.evaluate(
                            orig=image_in,
                            test=image_out,
                            method=method.__name__,
                            params=json.dumps(param_set),
                            dataset=random_noise.mode,
                            dataset_params=random_noise.params,
                        )
                        # Persist the evaluation.
                        self._add_evaluation(evaluation=ev)


# ------------------------------------------------------------------------------------------------ #
#                                RANDOM NOISE GEN                                                  #
# ------------------------------------------------------------------------------------------------ #
class RandomNoise:
    """Encapsulates the process of adding random noise to an image.

    Args:
        mode: (str): One of the modes in skimage.util.random_noise function.
        clip: (bool): If True, the output will be clipped to ensure that the values
            are in [-1,1]. Default = True
        mean: (float): For 'gaussian' and 'speckle' modes. Specifies the mean of the random
            distribution. Default = 0
        var: (float): The noise variance for 'gaussian' and 'speckle' noise. Default = 0.01
        amount (float): The proportion of pixels to replace with noise. Used for
            'salt', 'pepper', and 'salt & pepper' noise. Default = 0.05.
        svp (float): The proportion of salt vs pepper noise. Default = 0.5

    """

    def __init__(
        self,
        mode: str,
        clip: bool = True,
        mean: float = 0.0,
        var: float = 0.01,
        amount: float = 0.05,
        svp: float = 0.5,
    ) -> None:
        self._mode = mode
        self._clip = clip
        self._mean = mean
        self._var = var
        self._amount = amount
        self._svp = svp

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def params(self) -> json:
        d = {
            "clip": self._clip,
            "mean": self._mean,
            "var": self._var,
            "amount": self._amount,
            "svp": self._svp,
        }
        return json.dumps(d)

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        if self._mode == "gaussian":
            image_noisy = skimage.util.random_noise(
                image=image,
                mode=self._mode,
                clip=self._clip,
                mean=self._mean,
                var=self._var,
            )
        elif self._mode == "s&p":
            image_noisy = skimage.util.random_noise(
                image=image,
                mode=self._mode,
                clip=self._clip,
                amount=self._amount,
                salt_vs_pepper=self._svp,
            )
        elif self._mode == "speckle":
            image_noisy = skimage.util.random_noise(
                image=image,
                mode=self._mode,
                clip=self._clip,
                mean=self._mean,
                var=self._var,
            )
        return convert_uint8(img=image_noisy)
