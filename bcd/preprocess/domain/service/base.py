#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/imaging/domain/service/base.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 20th 2023 05:17:57 am                                                #
# Modified   : Friday October 20th 2023 05:24:39 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Domain Service Base Module"""
from abc import ABC, abstractmethod
from ..entity.image import Image


# ------------------------------------------------------------------------------------------------ #
class Transformer(ABC):
    @abstractmethod
    def transform(self, image: Image) -> Image:
        """Transforms an Image"""
