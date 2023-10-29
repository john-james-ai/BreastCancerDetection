#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/base.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 28th 2023 07:12:51 pm                                              #
# Modified   : Sunday October 29th 2023 03:34:12 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base Module for Image Preprocessing"""
from abc import abstractmethod

import numpy as np
from dependency_injector.wiring import Provide, inject

from bcd.container import BCDContainer
from bcd.core.base import Application
from bcd.core.image.entity import Image
from bcd.core.image.factory import ImageFactory
from bcd.dal.io.image import ImageIO
from bcd.dal.repo.uow import UoW


# ------------------------------------------------------------------------------------------------ #
class Transformer(Application):
    """Base Class for Image Transformers"""

    stage_id = None  # Overridden in subclasses

    @inject
    def __init__(
        self,
        task_id: str,
        io: ImageIO = Provide[BCDContainer.dal.io],
        image_factory: ImageFactory = Provide[BCDContainer.dal.image_factory],
        uow: UoW = Provide[BCDContainer.dal.uow],
    ) -> None:
        super().__init__()
        self._task_id = task_id
        self._io = io
        self._image_factory = image_factory
        self._uow = uow

    @property
    @abstractmethod
    def images_processed(self) -> int:
        """Returns the number of images processed by the Transformer."""

    @abstractmethod
    def execute(self) -> None:
        """Executes the transformation and returns a numpy array"""

    def create_image(self, case_id: str, pixel_data: np.ndarray) -> Image:
        """Creates an image using the ImageFactory

        Args:
            case_id (str): The case for the image.
            pixel_data (np.ndarray): The image pixel data.
        """
        return self._image_factory.create(
            case_id=case_id,
            stage_id=self.stage_id,
            pixel_data=pixel_data,
            transformer=self.__class__.__name__,
            task_id=self._task_id,
        )

    def read_pixel_data(self, filepath: str) -> np.ndarray:
        """Reads the image and returns the pixel data

        Args:
            filepath (str): The path to the image.
        """
        return self._io.read(filepath=filepath)

    def read_images(self, stage_id: int, transformer: str = None) -> dict:
        """Returns images for the given stage in a dict format, keyed by image uid.

        Args:
            stage_id (int): The stage identifier for which images are
                to be returned.
        """
        if transformer is None:
            _, images = self._uow.image_repo.get_by_stage(stage_id=stage_id)
        else:
            _, images = self._uow.image_repo.get_by_transformer(transformer=transformer)
        return images

    def read_image(self, uid: str) -> Image:
        """Reads an image from the repository

        Args:
            uid (str): The unique identifier for the image.
        """
        return self._uow.image_repo.get(uid=uid)

    def save_image(self, image: Image) -> None:
        """Saves an image to the repository.

        Args:
            image (Image): Image object.
        """
        self._uow.image_repo.add(image=image)
