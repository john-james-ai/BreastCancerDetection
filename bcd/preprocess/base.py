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
# Created    : Saturday October 21st 2023 03:43:04 pm                                              #
# Modified   : Sunday October 22nd 2023 12:03:15 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod

from bcd.manage_data.entity.image import Image

# ------------------------------------------------------------------------------------------------ #


class Task(ABC):
    """Defines the interface for image preprocessing tasks"""

    def execute(self, image: Image) -> Image:
        """Executes the task on the image"""
