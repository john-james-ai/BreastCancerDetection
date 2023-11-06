#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/denoise.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 31st 2023 04:45:05 am                                               #
# Modified   : Monday November 6th 2023 02:57:34 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Denoiser Task Module"""
import logging

from bcd.preprocess.image.flow.basetask import Task
from bcd.preprocess.image.method.basemethod import Method


# pylint: disable=useless-parent-delegation
# ------------------------------------------------------------------------------------------------ #
#                                DENOISER TASK                                                     #
# ------------------------------------------------------------------------------------------------ #
class DenoiserTask(Task):
    """Denoises images

    Args:
        task_params (DenoiserTaskParams): Parameters that control the task behavior.
        config (Config): The application configuration class
        io (ImageIO): The class responsible for image io
    """

    def __init__(
        self,
        method: Method,
        params: dict,
        batchsize: int = 16,
        instage_id: int = 0,
        outstage_id: int = 1,
    ) -> None:
        super().__init__(
            method=method,
            params=params,
            batchsize=batchsize,
            instage_id=instage_id,
            outstage_id=outstage_id,
        )

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def run(self) -> None:
        """Runs the Denoiser"""
        super().run()
