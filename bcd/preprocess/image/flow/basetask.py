#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/basetask.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 11:03:59 pm                                             #
# Modified   : Monday November 6th 2023 02:57:54 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the Interface for Task classes."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

from dependency_injector.wiring import Provide, inject
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from bcd.config import Config
from bcd.container import BCDContainer
from bcd.dal.io.image_reader import ImageReader
from bcd.dal.repo.uow import UoW
from bcd.image import ImageFactory
from bcd.preprocess.image.method.basemethod import Method


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Abstract base class for task objects."""

    @inject
    def __init__(
        self,
        instage_id: int,
        outstage_id: int,
        method: type[Method],
        params: dict,
        batchsize: int = 16,
        uow: UoW = Provide[BCDContainer.dal.uow],
        config: type[Config] = Config,
        reader: type[ImageReader] = ImageReader,
        factory: type[ImageFactory] = ImageFactory,
    ) -> None:
        self._instage_id = instage_id
        self._outstage_id = outstage_id
        self._method = method
        self._params = params
        self._batchsize = batchsize
        self._uow = uow
        self._config = config
        self._reader = reader
        self._factory = factory()

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def run(self) -> None:
        """Runs the Task"""
        condition = lambda df: df["stage_id"] == self._instage_id
        reader = self._reader(batchsize=self._batchsize, condition=condition)

        param_grid = self._get_paramgrid()

        for batch in tqdm(reader, desc="batch", total=reader.num_batches):
            for image_in in tqdm(batch, desc="image", total=len(batch)):
                for params in tqdm(param_grid, desc="param", total=len(param_grid)):
                    start = datetime.now()
                    pixel_data = self._method.execute(image_in.pixel_data, **params)
                    stop = datetime.now()
                    build_time = (stop - start).total_seconds()
                    image_out = self._factory.create(
                        case_id=image_in.case_id,
                        stage_id=self._outstage_id,
                        pixel_data=pixel_data,
                        method=self._method.__name__,
                        build_time=build_time,
                    )
                    self._uow.image_repo.add(image=image_out)

    def _get_paramgrid(self) -> list:
        """Returns a list of parameter sets based upon the param(grid)"""
        param_list = []
        param_grid = ParameterGrid(self._params)
        for param_set in param_grid:
            param_list.append(param_set)
        return param_list
