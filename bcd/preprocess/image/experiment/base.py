#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/experiment/base.py                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 11:03:59 pm                                             #
# Modified   : Wednesday November 29th 2023 10:28:19 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the Interface for Task classes."""
from __future__ import annotations

from abc import ABC, abstractmethod

from dependency_injector.wiring import Provide, inject

from bcd.config import Config
from bcd.container import BCDContainer
from bcd.dal.repo.uow import UoW
from bcd.image import Image
from bcd.preprocess.image.experiment.evaluate import Evaluation
from bcd.preprocess.image.flow.task import Task


# ------------------------------------------------------------------------------------------------ #
class Experiment(ABC):
    """Abstract base class for task objects.

    Args:
        uow (UoW): Unit of work class containing repositories.
        config (type[Config]): App config class
    """

    @inject
    def __init__(
        self,
        uow: UoW = Provide[BCDContainer.dal.uow],
        config: type[Config] = Config,
    ) -> None:
        self._uow = uow
        self._config = config

    @abstractmethod
    def run(self) -> None:
        """Runs the Task"""

    def _add_evaluation(self, evaluation: Evaluation) -> None:
        self._uow.eval_repo.add(evaluation=evaluation)

    def _add_task(self, task: Task) -> None:
        self._uow.task_repo.add(task=task)

    def _add_image(self, image: Image) -> None:
        self._uow.image_repo.add(image=image)
