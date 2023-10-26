#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/filter.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 23rd 2023 03:43:02 am                                                #
# Modified   : Thursday October 26th 2023 12:35:30 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Converts DICOM Data to PNG Format"""
from abc import abstractmethod
from dataclasses import dataclass

from tqdm import tqdm
import pandas as pd
import cv2
from dependency_injector.wiring import inject, Provide

from bcd.core.base import Stage
from bcd.core.image.entity import Image
from bcd.core.image.factory import ImageFactory
from bcd.preprocess.base import Preprocessor, Params
from bcd.core.image.repo import ImageRepo
from bcd.core.task.entity import Task
from bcd.container import BCDContainer


# ------------------------------------------------------------------------------------------------ #
@dataclass
class FilterParams(Params):
    kernel: int = 5


# ------------------------------------------------------------------------------------------------ #
class Filter(Preprocessor):
    MODULE = "bcd.preprocess.filter"
    STAGE = Stage(id=1)

    @inject
    def __init__(
        self,
        params: Params,
        task_id: str,
        image_repo: ImageRepo = Provide[BCDContainer.repo.image],
        image_factory: ImageFactory = Provide[BCDContainer.repo.factory],
    ) -> None:
        super().__init__(
            task_id=task_id,
            module=self.MODULE,
            stage=self.STAGE,
            params=params,
            image_repo=image_repo,
            image_factory=image_factory,
        )
        self._images_processed = 0
        self._kernel = params.kernel

    @property
    def images_processed(self) -> int:
        return self._images_processed

    def get_source_image_metadata(self) -> list:
        """Obtains the original images from the repository."""

        # Extract stage 0 images from repository
        condition = lambda df: df["stage_id"] == 0  # noqa
        return self._image_repo.get_meta(condition=condition)

    def process_images(self, image_metadata: pd.DataFrame) -> None:
        """Performs the filter operation on the images and stores in the repository.

        Args:
            images (list): List of Image objects.
        """
        for _, image in tqdm(image_metadata.iterrows(), total=image_metadata.shape[0]):
            image = self.read_image(id=image_metadata["id"].values[0])
            image = self.process_image(image)
            self.save_image(image=image)
            self._images_processed += 1

    @abstractmethod
    def process_image(self, image: Image) -> Image:
        """Performs filtering on the image."""


# ------------------------------------------------------------------------------------------------ #
class MeanFilter(Filter):
    def __init__(
        self,
        params: Params,
        task_id: str,
    ) -> None:
        super().__init__(params=params, task_id=task_id)
        self._kernel = params.kernel

    def process_image(self, image: Image) -> Image:
        pixel_data = cv2.blur(image.pixel_data, (self._kernel, self._kernel))
        return self.create_image(case_id=image.case_id, pixel_data=pixel_data)


# ------------------------------------------------------------------------------------------------ #
class MedianFilter(Filter):
    def __init__(
        self,
        params: Params,
        task_id: str,
    ) -> None:
        super().__init__(params=params, task_id=task_id)
        self._kernel = params.kernel

    def process_image(self, image: Image) -> Image:
        pixel_data = cv2.medianBlur(image.pixel_data, self._kernel)
        return self.create_image(case_id=image.case_id, pixel_data=pixel_data)


# ------------------------------------------------------------------------------------------------ #
class GaussianFilter(Filter):
    def __init__(
        self,
        params: Params,
        task_id: str,
    ) -> None:
        super().__init__(params=params, task_id=task_id)
        self._kernel = params.kernel

    def process_image(self, image: Image) -> Image:
        pixel_data = cv2.GaussianBlur(image.pixel_data, (self._kernel, self._kernel), 0)
        return self.create_image(case_id=image.case_id, pixel_data=pixel_data)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class FilterTask(Task):
    @classmethod
    def get_params(cls, params: str) -> Task:
        """Creates a Task object from  a dataframe"""
        return FilterParams.from_string(params=params)
