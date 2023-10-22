#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/application/filter.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 19th 2023 08:41:07 pm                                              #
# Modified   : Sunday October 22nd 2023 12:02:34 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Union, Tuple

from bcd.manage_data.structure.dataclass import DataClass
from bcd.preprocess.domain.service.filter import Filter

# Inject from container.
from bcd.manage_data.entity.image import Image


# ------------------------------------------------------------------------------------------------ #
class FilterParams(DataClass):
    kernel: Union[Tuple, int]
    diameter: int
    sigma_color: int
    sigma_scale: int


# ------------------------------------------------------------------------------------------------ #
class FilterCommand:
    def __init__(self, image: Image, filter: Filter, params: FilterParams) -> None:
        self._image = image
        self._filter = filter
        self._params = params

    def execute(self) -> None:
        # Execute the filter object
        return self._filter.filter(image=self._image, **self._params.as_dict())
