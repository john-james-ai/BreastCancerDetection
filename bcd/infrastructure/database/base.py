#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/infrastructure/database/base.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 09:56:27 am                                              #
# Modified   : Thursday October 26th 2023 11:15:38 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module provides basic database interface"""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging

import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class Database(ABC):
    """Abstract base class for databases."""

    def __init__(self) -> None:
        self._name = None
        self._engine = None
        self._connection = None
        self._transaction = None
        self._is_connected = False
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def name(self) -> str:
        """Returns the name of the database"""
        return self._name

    @property
    def is_connected(self) -> bool:
        """If connected, returns True; otherwise..."""
        return self._is_connected

    def __enter__(self) -> Database:
        """Enters a transaction block allowing multiple database operations to be performed as a unit."""
        self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # pragma: no cover
        """Special method takes care of properly releasing the object's resources to the operating system."""
        if exc_type is not None:
            try:
                self.rollback()
            except SQLAlchemyError as e:
                msg = f"Exception occurred.\nException type: {type[SQLAlchemyError]}\n{e}"
                self._logger.exception(msg)
                raise
            msg = f"Exception occurred.\nException type: {exc_type}\n{exc_value}\n{traceback}"
            self._logger.exception(msg)
            raise
        else:
            self.commit()
        self.close()

    @abstractmethod
    def connect(self, autocommit: bool = False):
        """Connect to an underlying database.

        Args:
            autocommit (bool): Sets autocommit mode. Default is False.
        """

    def begin(self):
        """Begins a transaction block."""
        try:
            self._transaction = self._connection.begin()
        except AttributeError:
            self.connect()
            self._transaction = self._connection.begin()
        except sqlalchemy.exc.InvalidRequestError:  # pragma: no cover
            self.close()
            self.connect()
            self._connection.begin()

    def in_transaction(self) -> bool:
        """Queries the autocommit mode and returns True if the connection is in transaction."""
        try:
            return self._connection.in_transaction()
        except SQLAlchemyError:  # pragma: no cover
            # ProgrammingError raised if connection is closed.
            return False

    def commit(self) -> None:
        """Saves pending database operations to the database."""
        try:
            self._connection.commit()
        except SQLAlchemyError as e:  # pragma: no cover
            msg = f"Exception occurred during connection commit.\n{e}"
            self._logger.exception(msg)
            raise

    def rollback(self) -> None:
        """Restores the database to the state of the last commit."""
        try:
            self._connection.rollback()
        except SQLAlchemyError as e:  # pragma: no cover
            msg = f"Exception occurred during connection rollback.\n{e}"
            self._logger.exception(msg)
            raise

    def close(self) -> None:
        """Closes the database connection."""
        try:
            self._connection.close()
            self._is_connected = False
        except SQLAlchemyError as e:  # pragma: no cover
            self._is_connected = False
            msg = f"Database connection close failed.\nException type: {type[e]}\n{e}"
            self._logger.exception(msg)
            raise

    def dispose(self) -> None:
        """Disposes the connection and releases resources."""
        try:
            self._engine.dispose()
            self._is_connected = False
        except SQLAlchemyError as e:  # pragma: no cover
            msg = f"Database connection close failed.\nException type: {type[e]}\n{e}"
            self._logger.exception(msg)
            raise

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
        try:
            return data.to_sql(
                tablename, con=self._connection, if_exists=if_exists, dtype=dtype, index=False
            )
        except SQLAlchemyError as e:  # pragma: no cover
            msg = f"Exception occurred during database insert.\nException type:{type[SQLAlchemyError]}\n{e}"
            self._logger.exception(msg)
            raise

    def update(self, query: str, params: dict = None) -> int:
        """Updates row(s) matching the query.

        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command

        Returns (int): Number of rows updated.
        """
        result = self.execute(query=query, params=params)
        return result.rowcount

    def delete(self, query: str, params: dict = None) -> int:
        """Deletes row(s) matching the query.

        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command

        Returns (int): Number of rows deleted.
        """
        result = self.execute(query=query, params=params)
        return result.rowcount

    def query(
        self, query: str, params: dict = (), dtype: dict = None, parse_dates: dict = None
    ) -> pd.DataFrame:
        """Fetches the next row of a query result set, returning a single sequence, or None if no more data
        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command
            dtype (dict): Dictionary mapping of column to data types
            parse_dates (dict): Dictionary of columns and keyword arguments for datetime parsing.

        Returns: Pandas DataFrame

        """
        return pd.read_sql(
            sql=sqlalchemy.text(query),
            con=self._connection,
            params=params,
            dtype=dtype,
            parse_dates=parse_dates,
        )

    def exists(self, query: str, params: dict = None) -> bool:
        """Returns True if a row matching the query and parameters exists. Returns False otherwise.
        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command

        """
        result = self.execute(query=query, params=params)
        result = result.fetchall()
        return result[0][0] != 0

    def execute(self, query: str, params: dict = ()) -> list:
        """Execute method reserved primarily for updates, and deletes, as opposed to queries returning data.

        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command

        Returns (int): Number of rows updated or deleted.

        """
        return self._connection.execute(statement=sqlalchemy.text(query), parameters=params)
