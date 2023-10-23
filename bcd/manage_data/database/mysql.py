#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /bcd/manage_data/database/mysql.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday April 10th 2023 09:50:40 pm                                                  #
# Modified   : Sunday October 22nd 2023 07:42:38 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""MySQL Database Module"""
from __future__ import annotations
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
import sqlalchemy
import subprocess
from time import sleep

from .base import Database
from .config import DatabaseConfig

# ------------------------------------------------------------------------------------------------ #
load_dotenv()


# ------------------------------------------------------------------------------------------------ #
class MySQLDatabase(Database):
    """MySQL Database Class
    Args:
        name (str): Name of database
    """

    def __init__(self, config: DatabaseConfig) -> None:
        super().__init__()
        self._config = config()
        self._name = self._config.name
        self._autocommit = self._config.autocommit
        self._connection_string = self._get_connection_string()
        self.connect()

    def connect(self) -> None:
        attempts = 0
        max_attempts = 3
        database_started = False
        while attempts < max_attempts:
            attempts += 1
            try:
                self._engine = sqlalchemy.create_engine(self._connection_string)
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

    def backup(self) -> str:
        """Performs a backup of the database to file"""
        directory = self._config.backup_directory
        filename = "appstore_" + datetime.now().strftime("%Y-%m-%d_T%H%M%S") + ".sql"
        filepath = os.path.abspath(os.path.join(directory, filename))
        os.makedirs(directory, exist_ok=True)
        try:
            with open(filepath, "w") as f:
                proc = subprocess.Popen(
                    [
                        "mysqldump",
                        "--user=%s" % self._config.username,
                        "--password=%s" % self._config.password,
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
            with open(filepath, "r") as f:
                command = [
                    "mysql",
                    "--user=%s" % self._config.username,
                    "--password=%s" % self._config.password,
                    self._name,
                ]
                proc = subprocess.Popen(command, stdin=f)
                stdout, stderr = proc.communicate()
        except ValueError as e:  # pragma: no cover
            msg = f"Suprocess POpen was called with invalid arguments.\n{e}"
            self._logger.exception(msg)
            raise

    def _get_connection_string(self) -> str:
        """Returns the connection string for the named database."""
        return f"mysql+pymysql://{self._config.username}:{self._config.password}@localhost/{self._name}"

    def _start_db(self) -> None:  # pragma: no cover
        subprocess.run([self._config.startup], shell=True)
