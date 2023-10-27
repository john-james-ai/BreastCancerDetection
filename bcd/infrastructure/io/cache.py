#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/infrastructure/io/cache.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 26th 2023 04:05:19 pm                                              #
# Modified   : Friday October 27th 2023 02:05:39 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Union
import logging
import pickle

import pandas as pd

from bcd.infrastructure.io.base import Cache
from bcd.infrastructure.io.image import ImageIO
from bcd.core.base import Entity


# ------------------------------------------------------------------------------------------------ #
class ImageCache(Cache):
    """Image Cache Manager"""

    def __init__(self) -> None:
        super().__init__()
        self._cache = {}
        self._registry = pd.DataFrame()
        self._io = ImageIO()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def registry(self) -> pd.DataFrame:
        """Returns the registry"""
        return self._registry

    def put(self, entity: Entity, write_through: bool = False) -> None:
        """Writes an image either to disc or to cache.

        Writes to disk if write_through is True. Otherwise, the entity
        is written to cache.

        Args:
            entity (Entity): Entity to be saved
            write_through (bool): Whether to write through to disk.
           to cache."""
        if write_through:
            self._io.write(pixel_data=entity.pixel_data, filepath=entity.filepath)
        else:
            key = pickle.dumps(entity.uid)
            value = pickle.dumps(entity)
            self._cache[key] = value
            self._registry = pd.concat([self._registry, entity.as_df()])





    def reset(self) -> None:
        """Disposes / resets the cache"""
        self._cache = {}
        self._registry = pd.DataFrame()

    def remove(self, uid) -> None:
        """Removes an image from the cache.

        Args:
             uid (str): The unique identifier for the entity.
        """
        try:
            key = pickle.dumps(uid)
            del self._cache[key]
            self._registry = self._registry.loc[self._registry['uid'] != uid]
        except KeyError:
            msg = f"Uid {uid} doesn't exist in the cache."
            self._logger.debug(msg)

    def save(self) -> None:
        """Saves all items to file"""
        for _, value in self._cache.items():
            entity = pickle.loads(value)
            self._io.write(pixel_data=entity.pixel_data, filepath=entity.filepath)
        self.reset()
