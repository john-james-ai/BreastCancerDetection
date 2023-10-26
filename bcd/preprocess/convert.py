#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/convert.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 09:59:41 pm                                                #
# Modified   : Thursday October 26th 2023 04:08:56 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Converts DICOM Data to PNG Format"""
import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import logging
from tqdm import tqdm

import pandas as pd
from dependency_injector.wiring import inject, Provide

from bcd.preprocess.base import Params, Preprocessor, Stage
from bcd.core.task.entity import Task
from bcd.core.image.factory import ImageFactory
from bcd.core.image.repo import ImageRepo
from bcd.container import BCDContainer

# ------------------------------------------------------------------------------------------------ #
load_dotenv()


# ------------------------------------------------------------------------------------------------ #
@dataclass()
class Stage0(Stage):
    id: int = 0


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ImageConverterParams(Params):
    frac: float = 0.01
    n_jobs: int = 6
    random_state: int = None


# ------------------------------------------------------------------------------------------------ #
class ImageConverter(Preprocessor):
    MODULE = "bcd.preprocess.convert"
    STAGE = Stage0()

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
        self._frac = params.frac
        self._n_jobs = params.n_jobs
        self._random_state = params.random_state
        self._images_processed = 0

    @property
    def images_processed(self) -> int:
        return self._images_processed

    def get_source_image_metadata(self) -> pd.DataFrame:
        """Performs multivariate stratified sampling to obtain a fraction of the raw images."""

        # Read the raw DICOM metadata
        df = pd.read_csv(os.getenv("DICOM_FILEPATH"))

        # Extract full mammogram images.
        self._images = df.loc[df["series_description"] == "full mammogram images"]

        # Define the stratum for stratified sampling
        stratum = ["image_view", "abnormality_type", "cancer", "assessment"]

        # Execute the sampling and obtain the case_ids
        df = self._images.groupby(by=stratum).sample(
            frac=self._frac, random_state=self._random_state
        )

        return df

    def process_images(self, image_metadata: pd.DataFrame) -> None:
        """Convert the images to PNG format and store in the repository.

        Args:
            image_metadata (pd.DataFrame): DataFrame containing image metadata.
        """
        for _, metadata in tqdm(image_metadata.iterrows(), total=image_metadata.shape[0]):
            self._process_image(image_metadata=metadata)

    def _process_image(self, image_metadata: pd.Series) -> None:
        """Reads image pixel data, creates a new image and persists it to the repository

        Args:
            image_metadata (pd.Series): Series containing image metadata
        """

        pixel_data = self.read_pixel_data(filepath=image_metadata["filepath"])
        image = self.create_image(case_id=image_metadata["case_id"], pixel_data=pixel_data)

        self.save_image(image=image)
        self._images_processed += 1


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ImageConverterTask(Task):
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Task:
        """Creates a Task object from  a dataframe"""
        application = cls._get_class(
            module_name=df["module"].values[0], class_name=df["name"].values[0]
        )

        params = ImageConverterParams.from_string(df["params"].values[0])

        return cls(
            id=df["id"].values[0],
            name=df["name"].values[0],
            application=application,
            mode=df["mode"].values[0],
            stage_id=df["stage_id"].values[0],
            stage=df["stage"].values[0],
            module=df["module"].values[0],
            params=params,
            images_processed=df["images_processed"].values[0],
            image_processing_time=df["image_processing_time"].values[0],
            started=df["started"].values[0].astype(datetime),
            ended=df["ended"].values[0].astype(datetime),
            duration=df["duration"].values[0],
            state=df["state"].values[0],
        )

    def run(self) -> None:
        """Executes the preprocessor."""
        self.start_task()
        app = self.application(params=self.params, task_id=self.id)
        try:
            app.execute()
            self.images_processed = app.images_processed
        except Exception as e:
            self.crash_task()
            logging.exception(e)
            raise
        else:
            self.end_task()
