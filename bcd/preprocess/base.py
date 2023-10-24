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
# Modified   : Tuesday October 24th 2023 04:34:53 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for the image preprocessing package."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod, abstractproperty
import logging
import json

import pandas as pd
import numpy as np

from bcd.manage_data import STAGES
from bcd.manage_data.repo.base import Repo
from bcd.manage_data.io.image import ImageIO
from bcd.manage_data.entity.image import Image, ImageFactory
from bcd.manage_data.structure.dataclass import DataClass

# ------------------------------------------------------------------------------------------------ #
load_dotenv()


# ------------------------------------------------------------------------------------------------ #
@dataclass()
class Stage(ABC):
    id: int
    name: str = None

    def __post_init__(self) -> None:
        try:
            self.name = STAGES[self.id]
        except KeyError:
            msg = f"{self.id} is an invalid stage id."
            logging.exception(msg)
            raise


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Params(DataClass):
    """Abstract base class for preprocessor parameters."""

    def as_string(self) -> str:
        d = self.as_dict()
        return json.dumps(d)


# ------------------------------------------------------------------------------------------------ #
class Preprocessor(ABC):
    """Defines the interface for image preprocessors"""

    def __init__(
        self,
        task_id: str,
        stage: Stage,
        params: Params,
        image_repo: Repo,
        image_factory: ImageFactory,
        io: type[ImageIO] = ImageIO,
    ) -> None:
        self._mode = os.getenv("MODE")
        self._task_id = task_id
        self._stage = stage
        self._params = params
        self._image_repo = image_repo
        self._image_factory = image_factory
        self._io = io()

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

        return self.__class__.__name__

    @property
    def task_id(self) -> int:
        """Stage id for the preprocessor"""
        return self._task_id

    @property
    def stage_id(self) -> int:
        """Stage id for the preprocessor"""
        return self._stage.id

    @property
    def stage(self) -> int:
        """Stage for the preprocessor"""
        return self._stage.name

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def mode(self) -> str:
        """Stage for the task."""
        return self._mode

    @property
    def params(self) -> str:
        return self._params

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
            stage_id=self._stage.id,
            stage=self._stage.name,
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
        condition = lambda df: df["id"] == id  # noqa
        self._image_repo.get(condition=condition)

    def save_image(self, image: Image) -> None:
        """Saves an image object to the repository."""
        self._image_repo.add(image=image)


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Abstract base class for tasks."""

    def __init__(self, preprocessor: Preprocessor, params: Params, stage: Stage) -> None:
        self._id = str(uuid4())
        self._started = None
        self._ended = None
        self._duration = None
        self._state = "PENDING"
        self._name = preprocessor.__class__.__name__
        self._stage_id = stage.id
        self._stage = stage.name
        self._params = params.as_string()
        self._mode = os.getenv("MODE")

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def stage_id(self) -> str:
        return self._stage_id

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def started(self) -> datetime:
        return self._started

    @property
    def ended(self) -> datetime:
        return self._ended

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def params(self) -> str:
        return self._params

    @property
    def state(self) -> str:
        return self._state

    @abstractproperty
    def images_processed(self) -> int:
        """Count of images processed"""

    @abstractmethod
    def run(self) -> None:
        """Runs the task."""

    def start_task(self) -> None:
        self._started = datetime.now()
        self._state = "STARTED"

    def end_task(self) -> None:
        self._ended = datetime.now()
        self._duration = (self._ended - self._started).total_seconds()
        self._state = "SUCCESS"

    def as_df(self) -> pd.DataFrame:
        d = {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "stage_id": self.stage_id,
            "stage": self.stage,
            "started": self.started,
            "ended": self.ended,
            "duration": self.duration,
            "images_processed": self.images_processed,
            "image_processing_time": self.images_processed / self.duration,
            "params": self.params,
            "state": self.state,
        }
        return pd.DataFrame(data=d, index=[0])
