#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/imaging/application/image.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 20th 2023 10:57:42 am                                                #
# Modified   : Friday October 20th 2023 11:10:05 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module that Provides Persistence Services via the Command Pattern"""
class CreateImageCommand()

# ------------------------------------------------------------------------------------------------ #
class CreateImageCommand:
    def __init__(self, image: Image, filter: Filter, params: FilterParams) -> None:
        self._image = image
        self._filter = filter
        self._params = params

    def execute(self) -> None:
        # Execute the filter object
        return self._filter.filter(image=self._image, **self._params.as_dict())
