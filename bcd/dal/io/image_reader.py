#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/io/image_reader.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 11:47:17 am                                              #
# Modified   : Sunday November 5th 2023 01:10:26 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import math
from typing import Callable, List, Union

import numpy as np
from dependency_injector.wiring import Provide, inject

from bcd.container import BCDContainer
from bcd.dal.repo.image import ImageRepo
from bcd.image import Image

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member


# ------------------------------------------------------------------------------------------------ #
#                                    IMAGE READER                                                  #
# ------------------------------------------------------------------------------------------------ #
class ImageReader:
    """Iterator that returns images individually or in batches.

    Args:
        batchsize (int): Size of batches to return. None is interpreted as a batch size of 1.
        condition (Callable): Lambda expression that will be used to select the data.
        repo (ImageRepo): Repository of images
    """

    @inject
    def __init__(
        self,
        batchsize: int = None,
        condition: Callable = None,
        repo: ImageRepo = Provide[BCDContainer.dal.image_repo],
    ) -> None:
        self._batchsize = batchsize
        self._condition = condition
        self._repo = repo
        self._index = 0
        self._image_metadata = None
        self._image_metadata_batches = None
        self._image_batch = []

    def __iter__(self) -> ImageReader:
        self._index = 0
        self._image_metadata = self._repo.get_meta(condition=self._condition).reset_index()
        if self._batchsize:
            n_batches = math.ceil(len(self._image_metadata) / self._batchsize)
            self._image_metadata_batches = np.array_split(self._image_metadata, n_batches)
        return self

    def __next__(self) -> Union[Image, List[Image]]:
        if self._batchsize:
            if self._index == len(self._image_metadata_batches):
                raise StopIteration
            return self._get_next_batch()
        else:
            if self._index == len(self._image_metadata):
                raise StopIteration
            return self._get_next_image()

    def _get_next_batch(self) -> List[Image]:
        images = []
        batch = self._image_metadata_batches[self._index]
        for _, image_meta in batch.iterrows():
            image = self._repo.get(uid=image_meta["uid"])
            images.append(image)
        self._index += 1
        return images

    def _get_next_image(self) -> Image:
        uid = self._image_metadata.iloc[self._index]["uid"]
        self._index += 1
        return self._repo.get(uid=uid)
