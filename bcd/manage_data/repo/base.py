#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/manage_data/repo/base.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 07:44:00 pm                                                #
# Modified   : Sunday October 22nd 2023 08:57:35 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module provides basic database interface"""
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, Any

from bcd.manage_data.entity.base import Entity
from bcd.manage_data.database.base import Database


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    """Provides base class for all repositories classes.

    Args:
        name (str): Repository name. This will be the name of the underlying database table.
        database(Database): Database containing data to access.
    """

    @abstractproperty
    def database(self) -> Database:
        """Returns the underlying database."""

    @abstractmethod
    def add(self, entity: Entity) -> None:
        """Adds an entity to the repository

        Args:
            entity (Entity): An entity object
        """

    @abstractmethod
    def get(self, condition: Callable, *args, **kwargs) -> Any:
        """Gets the entities that match the given condition

        Args:
            condition (Callable): A lambda conditional expression used to subset the data
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
    def delete(self, condition: Callable) -> None:
        """Deletes the entity or entities matching condition.

        Args:
            condition (Callable): Lambda expression subsetting the data.

        """
