#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/infrastructure/repo/base.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 07:44:00 pm                                                #
# Modified   : Thursday October 26th 2023 04:18:13 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module provides basic database interface"""
from abc import ABC, abstractmethod
from typing import Callable, Any

from bcd.core.base import Entity


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    """Provides base class for all repositories classes.

    Args:
        name (str): Repository name. This will be the name of the underlying database table.
        database(Database): Database containing data to access.
    """

    @abstractmethod
    def add(self, entity: Entity) -> None:
        """Adds an entity to the repository

        Args:
            entity (Entity): An entity object
        """

    @abstractmethod
    def get(self, id: str) -> Any:
        """Gets an an entity by its identifier.

        Args:
            id (str): Entity identifier.
        """

    @abstractmethod
    def exists(self, id: str) -> bool:
        """Evaluates existence of an entity by identifier.

        Args:
            id (str): Entity UUID

        Returns:
            Boolean indicator of existence.
        """

    @abstractmethod
    def count(self, condition: Callable = None) -> int:  # noqa
        """Counts the entities matching the criteria. Counts all entities if id is None.

        Args:
            condition (Callable): A lambda expression used to subset the data.

        Returns:
            Integer number of rows matching criteria
        """

    @abstractmethod
    def delete(self, id: str) -> None:
        """Deletes the entity or entities matching condition.

        Args:
            id (str): Entity identifier.

        """
