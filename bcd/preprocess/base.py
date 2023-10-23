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
# Modified   : Monday October 23rd 2023 01:52:39 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base module for the image preprocessing package."""
from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

from bcd.manage_data.structure.dataclass import DataClass


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Defines the interface for image preprocessing tasks"""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractproperty
    def stage_id(self) -> int:
        """Stage id for the task"""

    @abstractproperty
    def stage(self) -> str:
        """Stage for the task."""

    @abstractproperty
    def taskrun(self) -> TaskRun:
        """Returns the TaskRun object."""

    @abstractmethod
    def execute(self):
        """Executes the task"""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskRun(DataClass):
    id: str
    task: str
    mode: str
    stage_id: int
    stage: str
    started: datetime = None
    ended: datetime = None
    duration: float = None
    images_processed: int = None
    image_processing_time: float = None
    success: bool = None
    params: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))

    def start(self) -> None:
        self.started = datetime.now()

    def end(self) -> None:
        self.ended = datetime.now()
        self.duration = (self.ended - self.started).total_seconds()
        self.image_processing_time = self.duration / self.images_processed
        self.success = True
