#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 10:17:41 pm                                                #
# Modified   : Thursday October 26th 2023 12:19:10 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for the image preprocessing package."""
from __future__ import annotations
from abc import abstractmethod, abstractproperty
import logging

import pandas as pd
import numpy as np

from bcd.core.base import Repo, Application, Params, Stage
from bcd.infrastructure.io.image import ImageIO
from bcd.core.image.entity import Image
from bcd.core.image.factory import ImageFactory


# ------------------------------------------------------------------------------------------------ #
class Preprocessor(Application):
    """Defines the interface for image preprocessors"""

    def __init__(
        self,
        task_id: str,
        module: str,
        stage: Stage,
        params: Params,
        image_repo: Repo,
        image_factory: ImageFactory,
        io: type[ImageIO] = ImageIO,
    ) -> None:
        self._task_id = task_id
        self._module = module
        self._stage_id = stage.id
        self._stage = stage.name
        self._params = params
        self._image_repo = image_repo
        self._image_factory = image_factory
        self._io = io()

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

        return self.__class__.__name__

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def params(self) -> str:
        return self._params

    @property
    def module(self) -> str:
        return self._module

    @property
    def stage_id(self) -> int:
        return self._stage_id

    @property
    def stage(self) -> str:
        return self._stage

    @abstractproperty
    def images_processed(self) -> int:
        """Count of images processed"""

    @abstractmethod
    def get_source_image_metadata(self) -> pd.DataFrame:
        """Obtains the images or image metadata to be processed."""

    @abstractmethod
    def process_images(self, image_metadata: pd.DataFrame) -> None:
        """Processes the images

        Args:
            images (pd.DataFrame): DataFrame containing image
                metadata
        """

    def execute(self):
        """Executes the task"""

        image_metadata = self.get_source_image_metadata()

        self.process_images(image_metadata=image_metadata)

    def create_image(self, case_id: str, pixel_data: np.ndarray) -> Image:
        """Creates an image for a given case

        Args:
            case_id (str): Case identifier
            pixel_data (np.ndarray): Pixel data in numpy array format.

        Returns
            Image object
        """
        return self._image_factory.create(
            case_id=case_id,
            stage_id=self.stage_id,
            pixel_data=pixel_data,
            preprocessor=self.name,
            task_id=self.task_id,
        )

    def read_pixel_data(self, filepath) -> np.ndarray:
        """Reads an image pixel data from a file.

        Args:
            filepath (str): Path to image file.
        """
        return self._io.read(filepath=filepath)

    def read_image(self, id: str) -> Image:
        """Reads an image object from the repository."""
        return self._image_repo.get(id=id)

    def save_image(self, image: Image) -> None:
        """Saves an image object to the repository."""
        self._image_repo.add(image=image)
