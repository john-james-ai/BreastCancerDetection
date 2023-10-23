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
# Modified   : Sunday October 22nd 2023 11:33:24 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Converts DICOM Data to PNG Format"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

import pandas as pd
from dependency_injector.wiring import inject, Provide

from bcd.preprocess.base import Task, TaskRun, TaskParams
from bcd.manage_data.repo.base import Repo
from bcd.manage_data.entity.image import ImageFactory
from bcd.container import BCDContainer
# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #
class ImageConverter(Task):
    @inject
    def __init__(self, frac: float = 0.1, random_state: int = None, image_repo: Repo = Provide[BCDContainer], taskrun_repo: Repo, image_factory: ImageFactory) -> None:
        df = pd.read_csv(os.getenv('DICOM_FILEPATH'))
        self._cases = df.loc[df['series_description'] == 'full mammogram images']
        self._frac = frac
        self._random_state = random_state
        self._image_repo = image_repo
        self._taskrun_repo = taskrun_repo
        self._image_factory = ImageFactory