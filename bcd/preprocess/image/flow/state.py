#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/state.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 31st 2023 05:16:39 am                                               #
# Modified   : Saturday November 4th 2023 05:53:16 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
from dataclasses import dataclass
from datetime import datetime

# ------------------------------------------------------------------------------------------------ #
STAGES = {
    0: "Convert",
    1: "Denoise",
    2: "Artifact Removed",
    3: "Pectoral Removed",
    4: "Enhanced",
    5: "ROI Segmented",
    6: "Augmented",
    7: "Reshaped",
}


# ------------------------------------------------------------------------------------------------ #
@dataclass
class State:
    """A value object encapsulating the state of a Task or Job object."""

    images_processed: int = 0
    image_processing_time: float = 0
    started: datetime = None
    stopped: datetime = None
    duration: float = 0
    status: str = "PENDING"

    def start(self) -> None:
        self.started = datetime.now()
        self.status = "IN-PROGRESS"

    def count(self) -> None:
        self.images_processed += 1

    def stop(self) -> None:
        self.stopped = datetime.now()
        self.duration = (self.stopped - self.started).total_seconds()
        self.image_processing_time = self.images_processed / self.duration
        self.status = "COMPLETED"


# ------------------------------------------------------------------------------------------------ #
@dataclass()
class Stage:
    """Encapsulates a stage in the preprocessing and modeling phases."""

    uid: int
    name: str = None

    def __post_init__(self) -> None:
        try:
            self.name = STAGES[self.uid]
        except KeyError as e:
            msg = f"{self.uid} is an invalid stage id."
            logging.exception(msg)
            raise ValueError(msg) from e
