#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/infrastructure/io/base.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 26th 2023 03:57:11 pm                                              #
# Modified   : Friday October 27th 2023 02:04:54 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Cache Module for Delayed Writing in Repositories"""
from abc import ABC, abstractmethod
from bcd.core.base import Entity


# ------------------------------------------------------------------------------------------------ #
class Cache(ABC):
    """Abstract base dlass for cache objects."""

    @abstractmethod
    def put(self, entity: Entity) -> None:
        """Puts an entity in cache.

        Args
            entity (Entity): An entity to write to cache.
        """

    @abstractmethod
    def save(self) -> None:
        """Saves all entities to file."""


    @abstractmethod
    def reset(self) -> None:
        """Disposes / resets the cache"""

