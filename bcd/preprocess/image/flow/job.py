#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/job.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 25th 2023 11:03:59 pm                                             #
# Modified   : Tuesday October 31st 2023 05:13:14 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the Interface for Task classes."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from bcd.config import Config
from bcd.core.base import DataClass


# ------------------------------------------------------------------------------------------------ #
#                                          JOB                                                     #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class JobDTO(DataClass):
    """Encapsulates the command for a pipeline job, providing iteration over Task objects."""

    name: str
    uid: str = None
    mode: str = None
    n_tasks: int = 0

    def __post_init__(self) -> None:
        self.uid = str(uuid4())
        self.mode = Config.get_mode()
