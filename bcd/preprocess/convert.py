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
# Modified   : Monday October 23rd 2023 03:24:17 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Converts DICOM Data to PNG Format"""
import os
from dotenv import load_dotenv
from uuid import uuid4
from joblib import Parallel, delayed

from tqdm import tqdm
import pandas as pd
from dependency_injector.wiring import inject, Provide

from bcd.preprocess.base import Task, TaskRun
from bcd.manage_data.repo.image import ImageRepo
from bcd.manage_data.repo.task import TaskRunRepo
from bcd.manage_data.entity.image import ImageFactory
from bcd.container import BCDContainer
from bcd.manage_data import STAGES

# ------------------------------------------------------------------------------------------------ #
load_dotenv()


# ------------------------------------------------------------------------------------------------ #
class ImageConverter(Task):
    @inject
    def __init__(
        self,
        frac: float = 0.1,
        image_repo: ImageRepo = Provide[BCDContainer.repo.image],
        taskrun_repo: TaskRunRepo = Provide[BCDContainer.repo.taskrun],
        image_factory: ImageFactory = Provide[BCDContainer.repo.factory],
        n_jobs: int = 6,
        random_state: int = None,
    ) -> None:
        df = pd.read_csv(os.getenv("DICOM_FILEPATH"))
        self._images = df.loc[df["series_description"] == "full mammogram images"]
        self._frac = frac
        self._n_jobs = n_jobs
        self._random_state = random_state
        self._image_repo = image_repo
        self._taskrun_repo = taskrun_repo
        self._image_factory = image_factory
        self._taskrun_id = str(uuid4())
        self._stage_id = 0
        self._stage = STAGES[0]
        self._mode = os.getenv("MODE")
        self._params = {"frac": self._frac}

        self._taskrun = TaskRun(
            id=self._taskrun_id,
            task=self.name,
            mode=self._mode,
            stage_id=self._stage_id,
            stage=self._stage,
            params=self._params,
        )

    @property
    def stage_id(self) -> int:
        return self._stage_id

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def taskrun(self) -> TaskRun:
        return self._taskrun

    def execute(self) -> None:
        images = self._get_images()
        self._taskrun.images_processed = len(images)
        self._taskrun.start()
        for image in tqdm(images):
            self._image_repo.add(image)
        self._taskrun.end()

    def _get_images(self) -> list:
        images = []
        stratum = ["image_view", "abnormality_type", "cancer", "assessment"]
        df = self._images.groupby(by=stratum).sample(frac=self._frac)
        case_ids = list(df["case_id"])

        images = Parallel(n_jobs=self._n_jobs)(
            delayed(self._image_factory.from_case)(
                case_id=case_id, stage_id=0, task=self.name, taskrun_id=self._taskrun_id
            )
            for case_id in tqdm(case_ids)
        )
        return images
