#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/orchestration/job.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 27th 2023 04:09:45 pm                                                #
# Modified   : Sunday October 29th 2023 03:13:26 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Job Module: Defining the the abstract base class command object for pipeline jobs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

from bcd.config import Config
from bcd.core.base import Entity
from bcd.core.orchestration.task import Task


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Job(Entity):
    """Encapsulates the command for a pipeline job, providing iteration over Task objects."""

    name: str
    uid: str = None
    mode: str = None
    tasks: List[Task] = field(default_factory=list)
    n_tasks: int = 0

    def __post_init__(self) -> None:
        self.uid = str(uuid4())
        self.mode = Config.get_mode()

    def add_task(self, task: Task) -> None:
        """Adds a Task object to the Job"""
        task.job_id = self.uid
        self.tasks.append(task)
        self.n_tasks += 1
