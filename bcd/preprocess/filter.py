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
# Modified   : Tuesday October 24th 2023 03:23:31 am                                               #
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

from bcd.manage_data.entity.image import Image, ImageFactory
from bcd.preprocess.base import Preprocessor, Params, Stage
from bcd.manage_data.repo.image import ImageRepo
from bcd.container import BCDContainer


# ------------------------------------------------------------------------------------------------ #
@dataclass
class FilterParams(Params):
    kernel: int = 5


# ------------------------------------------------------------------------------------------------ #


@dataclass()
class Stage1(Stage):
    id: int = 1


# ------------------------------------------------------------------------------------------------ #
class Filter(Preprocessor):
    __STAGE_ID = 1

    @inject
    def __init__(
        self,
        params: Params,
        task_id: str,
        stage: Stage = Stage1(),
        image_repo: ImageRepo = Provide[BCDContainer.repo.image],
        image_factory: ImageFactory = Provide[BCDContainer.repo.factory],
    ) -> None:
        super().__init__(
            task_id=task_id,
            stage=stage,
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
        return self._image_repo.get(condition=condition)

    def process_images(self, image_metadata: pd.DataFrame) -> None:
        """Performs the filter operation on the images and stores in the repository.

        Args:
            images (list): List of Image objects.
        """
        for _, image in tqdm(image_metadata.iterrows()):
            image = self.process_image(image)
            self.save_image(image=image)

    @abstractmethod
    def process_image(self, image: Image) -> Image:
        """Performs filtering on the image."""


# ------------------------------------------------------------------------------------------------ #
class MeanFilter(Filter):
    __STAGE_ID = 1

    def __init__(
        self,
        kernel: int = 5,
    ) -> None:
        super().__init__()
        self._kernel = kernel

    def process_image(self, image: Image) -> Image:
        pixel_data = cv2.blur(image.pixel_data, (self._kernel, self._kernel))
        return self.create_image(case_id=image.case_id, pixel_data=pixel_data)


# ------------------------------------------------------------------------------------------------ #
class MedianFilter(Filter):
    __STAGE_ID = 1

    def __init__(
        self,
        kernel: int = 5,
    ) -> None:
        super().__init__()
        self._kernel = kernel

    def process_image(self, image: Image) -> Image:
        pixel_data = cv2.medianBlur(image.pixel_data, self._kernel)
        return self.create_image(case_id=image.case_id, pixel_data=pixel_data)


# ------------------------------------------------------------------------------------------------ #
class GaussianFilter(Filter):
    __STAGE_ID = 1

    def __init__(
        self,
        kernel: int = 5,
    ) -> None:
        super().__init__()
        self._kernel = kernel

    def process_image(self, image: Image) -> Image:
        pixel_data = cv2.GaussianBlur(image.pixel_data, (self._kernel, self._kernel), 0)
        return self.create_image(case_id=image.case_id, pixel_data=pixel_data)
