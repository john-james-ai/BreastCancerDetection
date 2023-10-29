#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/database/mysql.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 09:56:27 am                                              #
# Modified   : Sunday October 29th 2023 02:38:35 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module provides basic database interface"""
from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime
from time import sleep

import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError

from bcd.config import Config
from bcd.dal.database.base import Database

# ------------------------------------------------------------------------------------------------ #
load_dotenv()


# ------------------------------------------------------------------------------------------------ #
class MySQLDatabase(Database):
    """Abstract base class for databases."""

    def __init__(self, config: type[Config] = Config) -> None:
        super().__init__()
        self._name = config.get_name()
        self._user_name = config.get_username()
        self._pwd = config.get_password()
        self._startup_script = config.get_startup()
        self._backup_directory = config.get_backup_directory()
        self._autocommit = config.get_autocommit()
        self._timeout = config.get_timeout()
        self._max_attempts = config.get_max_attempts()
        self._engine = None
        self._connection = None
        self._transaction = None
        self._is_connected = False

        self._connection_string = self._get_connection_string()
        self.connect()

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
        """Enters a transaction block."""
        self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # pragma: no cover
        """Special method takes care of properly releasing the object's resources.

        Args:
            exc_type (str):  Exception type
            exc_value (str): Exception value
            traceback (str): Exception traceback
        """
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

    def connect(self):
        """Connect to an underlying database."""
        attempts = 0
        database_started = False
        while attempts < self._max_attempts:
            attempts += 1
            try:
                self._engine = sqlalchemy.create_engine(
                    self._connection_string,
                    connect_args={"read_timeout": int(self._timeout)},
                )
                self._connection = self._engine.connect()
                if self._autocommit is True:
                    self._connection.execution_options(isolation_level="AUTOCOMMIT")
                else:
                    self._connection.execution_options(isolation_level="READ UNCOMMITTED")
                self._is_connected = True
                database_started = True

            except SQLAlchemyError as e:  # pragma: no cover
                self._is_connected = False
                if not database_started:
                    msg = "Database is not started. Starting database..."
                    self._logger.info(msg)
                    self._start_db()
                    database_started = True
                    sleep(3)
                else:
                    msg = f"Database connection failed.\nException type: {type[e]}\n{e}"
                    self._logger.exception(msg)
                    raise
            else:
                return self

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
            msg = f"Exception occurred during database insert.\nException \
                type:{type[SQLAlchemyError]}\n{e}"
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
        """Fetches the next row of a result set, returning a sequence, or None if no more data
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
        """Execute method reserved primarily for updates, and deletes, as opposed to queries.

        Args:
            query (str): The SQL command
            params (dict): Parameters for the SQL command

        Returns (int): Number of rows updated or deleted.

        """
        return self._connection.execute(statement=sqlalchemy.text(query), parameters=params)

    def backup(self) -> str:
        """Performs a backup of the database to file"""

        filename = "bcd_" + datetime.now().strftime("%Y-%m-%d_T%H%M%S") + ".sql"
        filepath = os.path.abspath(os.path.join(self._backup_directory, filename))
        os.makedirs(self._backup_directory, exist_ok=True)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                proc = subprocess.Popen(
                    [
                        "mysqldump",
                        "--user=%s" % self._user_name,
                        "--password=%s" % self._pwd,
                        "--add-drop-database",
                        "--skip-add-drop-table",
                        "--databases",
                        self._name,
                    ],
                    stdout=f,
                )
                proc.communicate()
        except ValueError as e:  # pragma: no cover
            msg = f"Suprocess POpen was called with invalid arguments.\n{e}"
            self._logger.exception(msg)
            raise
        else:
            return filepath

    def restore(self, filepath: str) -> None:
        """Restores the database from a backup file.

        Args:
            filepath (str): The backup file on the local file system.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                command = [
                    "mysql",
                    "--user=%s" % self._user_name,
                    "--password=%s" % self._pwd,
                    self._name,
                ]
                proc = subprocess.Popen(command, stdin=f)
                proc.communicate()
        except ValueError as e:  # pragma: no cover
            msg = f"Suprocess POpen was called with invalid arguments.\n{e}"
            self._logger.exception(msg)
            raise

    def _get_connection_string(self) -> str:
        """Returns the connection string for the named database."""
        return f"mysql+pymysql://{self._user_name}:{self._pwd}@localhost/{self._name}"

    def _start_db(self) -> None:  # pragma: no cover
        subprocess.run([self._startup_script], shell=True, check=True)
