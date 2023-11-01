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
# Modified   : Wednesday November 1st 2023 01:49:19 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Defines the interface for Repository objects."""
from abc import ABC, abstractmethod
from typing import Callable

from bcd import Entity
from bcd.dal.database.base import Database


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    """Provides base class for all repositories classes.

    Args:
        database(Database): Database containing data to access.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    @property
    def database(self) -> Database:
        return self._database

    @abstractmethod
    def add(self, entity: Entity) -> None:
        """Adds an entity to the repository

        Args:
            entity (Entity): An entity object
        """

    @abstractmethod
    def get(self, **kwargs) -> Entity:
        """Gets an an entity by its identifier.

        Args:
            uid (str): Entity identifier.
        """

    @abstractmethod
    def exists(self, **kwargs) -> bool:
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
    def delete(self, **kwargs) -> None:
        """Deletes the entity or entities matching condition.

        Args:
            uid (str): Entity identifier.

        """

    def connect(self) -> None:
        self._database.connect()

    def close(self) -> None:
        self._database.close()
