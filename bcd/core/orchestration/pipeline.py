#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/orchestration/pipeline.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 27th 2023 04:07:36 pm                                                #
# Modified   : Friday October 27th 2023 04:07:36 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod

# ------------------------------------------------------------------------------------------------ #
class Pipeline(ABC):
    """Abstract base class for pipelines that execute jobs as a sequence of tasks."""
    @abstractmethod
    def add_job(self, job: Job) -> None:
    @abstractmethod
    def run(self) -> None:
        """Executes the pipeline job"""
    