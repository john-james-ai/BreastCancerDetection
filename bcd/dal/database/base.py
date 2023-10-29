#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/database/base.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 09:56:27 am                                              #
# Modified   : Sunday October 29th 2023 12:55:26 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module provides basic database interface"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class Database(ABC):
    """Abstract base class for databases."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the database"""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """If connected, returns True; otherwise..."""

    @abstractmethod
    def __enter__(self) -> Database:
        """Enters a transaction block."""

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:  # pragma: no cover
        """Special method takes care of properly releasing the object's resources.

        Args:
            exc_type (str):  Exception type
            exc_value (str): Exception value
            traceback (str): Exception traceback
        """

    @abstractmethod
    def connect(self):
        """Connect to an underlying database.

        Args:
            autocommit (bool): Sets autocommit mode. Default is False.
        """

    @abstractmethod
    def begin(self):
        """Begins a transaction block."""

    @abstractmethod
    def in_transaction(self) -> bool:
        """Queries the autocommit mode and returns True if the connection is in transaction."""

    @abstractmethod
    def commit(self) -> None:
        """Saves pending database operations to the database."""

    @abstractmethod
    def rollback(self) -> None:
        """Restores the database to the state of the last commit."""

    @abstractmethod
    def close(self) -> None:
        """Closes the database connection."""

    @abstractmethod
    def dispose(self) -> None:
        """Disposes the connection and releases resources."""

    @abstractmethod
    def insert(
        self, data: pd.DataFrame, tablename: str, dtype: dict = None, if_exists: str = "append"
    ) -> int:
        """Inserts data in pandas DataFrame format into the designated table.

        Note: This method uses pandas to_sql method. If not in transaction, inserts are
        autocommitted and rollback has no effect. Transaction behavior is extant
        after a begin() or through the use of the context manager.

        Args:
            data (pd.DataFrame): DataFrame containing the data to add to the designated table.
            tablename (str): The name of the table in the database. If the table does not
                exist, it will be created.
            dtype (dict): Dictionary of data types for columns.
            if_exists (str): Action to take if table already exists. Valid values
                are ['append', 'replace', 'fail']. Default = 'append'

        Returns: Number of rows inserted.
        """

    @abstractmethod
    def update(self, query: str, params: dict = None) -> int:
        """Updates row(s) matching the query.

        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command

        Returns (int): Number of rows updated.
        """

    @abstractmethod
    def delete(self, query: str, params: dict = None) -> int:
        """Deletes row(s) matching the query.

        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command

        Returns (int): Number of rows deleted.
        """

    @abstractmethod
    def query(
        self, query: str, params: dict = (), dtype: dict = None, parse_dates: dict = None
    ) -> pd.DataFrame:
        """Fetches the next row of a result set, returning a sequence, or None if no more data
        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command
            dtype (dict): Dictionary mapping of column to data types
            parse_dates (dict): Dictionary of columns and keyword arguments for datetime parsing.

        Returns: Pandas DataFrame

        """

    @abstractmethod
    def exists(self, query: str, params: dict = None) -> bool:
        """Returns True if a row matching the query and parameters exists. Returns False otherwise.
        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command

        """

    @abstractmethod
    def execute(self, query: str, params: dict = ()) -> list:
        """Execute method reserved primarily for updates, and deletes, as opposed to queries.

        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command

        Returns (int): Number of rows updated or deleted.

        """

    @abstractmethod
    def backup(self) -> str:
        """Performs a backup of the database to file"""

    @abstractmethod
    def restore(self, filepath: str) -> None:
        """Restores the database from a backup file.

        Args:
            filepath (str): The backup file on the local file system.
        """
