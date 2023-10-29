#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/repo/base.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 28th 2023 08:24:06 pm                                              #
# Modified   : Sunday October 29th 2023 02:22:35 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the interface for Repository objects."""
from abc import ABC, abstractmethod
from typing import Callable

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
    def get(self, uid: str) -> Entity:
        """Gets an an entity by its identifier.

        Args:
            uid (str): Entity identifier.
        """

    @abstractmethod
    def exists(self, uid: str) -> bool:
        """Evaluates existence of an entity by identifier.

        Args:
            uid (str): Entity UUID

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
    def delete(self, uid: str) -> None:
        """Deletes the entity or entities matching condition.

        Args:
            uid (str): Entity identifier.

        """