#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/base.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday December 29th 2023 12:22:16 am                                               #
# Modified   : Friday December 29th 2023 01:47:51 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base Module for Data Access Layer"""
from abc import ABC, abstractmethod
from typing import Any


# ------------------------------------------------------------------------------------------------ #
class Database(ABC):
    """Database Abstraction"""

    @abstractmethod
    def create(self, *args, **kwargs) -> None:
        """Creates an instance in the database"""

    @abstractmethod
    def read(self, *args, **kwargs) -> Any:
        """Reads from the database"""

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Updates the database"""

    @abstractmethod
    def delete(self, *args, **kwargs) -> None:
        """Deletes an instance from the database."""

    @abstractmethod
    def exists(self, name) -> bool:
        """Determines whether a named Dataset object exists."""

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        """Saves an instance either through creation or update."""
