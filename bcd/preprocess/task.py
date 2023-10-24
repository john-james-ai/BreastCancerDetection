#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/task.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 24th 2023 12:32:32 am                                               #
# Modified   : Tuesday October 24th 2023 02:50:11 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #

from bcd.preprocess.base import Preprocessor, Params, Stage, Task


# ------------------------------------------------------------------------------------------------ #
class PreprocessorTask(Task):
    def __init__(self, preprocessor: type[Preprocessor], stage: Stage, params: Params) -> None:
        super().__init__(preprocessor=preprocessor, stage=stage)
        self._preprocessor = preprocessor
        self._params = params
        self._stage = stage
        self._images_processed = 0

    def run(self) -> None:
        """Executes the preprocessor."""
        self.start_task()
        self._preprocessor(params=self._params, task_id=self.id)
        self._preprocessor.execute()
        self.end_task()
