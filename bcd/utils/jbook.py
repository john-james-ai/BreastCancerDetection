#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /bcd/utils/jbook.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 31st 2023 10:11:34 pm                                               #
# Modified   : Thursday August 31st 2023 10:17:00 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from glob import glob
import logging

import jupytext

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
class DocConverter:
    """Converts documents between jupyter and myst formats"""

    __jbook_module = "jbook"
    __jbook_search_path = "jbook/**/*.md"
    __notebook_module = "notebooks"
    __notebook_search_path = "notebooks/content/**/*.ipynb"

    def __init__(
        self,
        jbook_module: str = None,
        jbook_search_path: str = None,
        notebook_module: str = None,
        notebook_search_path: str = None,
    ) -> None:
        self._jbook_module = jbook_module or self.__jbook_module
        self._jbook_search_path = jbook_search_path or self.__jbook_search_path
        self._notebook_module = notebook_module or self.__notebook_module
        self._notebook_search_path = notebook_search_path or self.__notebook_search_path

    def get_jbook_filepaths(self) -> list:
        """Returns a list of jbook filepaths containing convertable content."""
        return glob(pathname=self._jbook_search_path, root_dir=os.getcwd(), recursive=True)

    def get_notebook_filepaths(self) -> list:
        """Returns a list of notebook filepaths containing convertable content."""
        return glob(pathname=self._notebook_search_path, root_dir=os.getcwd(), recursive=True)

    def to_notebook(self, force: bool = False) -> None:
        """Converts myst markdown files on the jbook search path to jupyter notebooks on the notebook search path.

        Args:
            force (str): If True, conversion will overwrite files in the destination without
                prompting for confirmation. Otherwise, a confirmation prompt is presented before
                overwriting existing notebooks.
        """
        sources = self.get_jbook_filepaths()
        dests = [fp.replace(".md", ".ipynb") for fp in sources]
        dests = [dest.replace(self._jbook_module, self._notebook_module) for dest in dests]

        for source, dest in zip(sources, dests):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and not force:
                if self._user_approves(dest=dest):
                    self._to_notebook(source=source, dest=dest)
            else:
                self._to_notebook(source=source, dest=dest)

    def to_jbook(self, force: bool = False) -> None:
        """Converts jupyter notebooks on the notebook search path to myst markdown on the jbook search path.

        Args:
            force (str): If True, conversion will overwrite files in the destination without
                prompting for confirmation. Otherwise, a confirmation prompt is presented before
                overwriting existing notebooks.
        """
        sources = self.get_notebook_filepaths()
        dests = [fp.replace(".ipynb", ".md") for fp in sources]
        dests = [dest.replace(self._notebook_module, self._jbook_module) for dest in dests]

        for source, dest in zip(sources, dests):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and not force:
                if self._user_approves(dest=dest):
                    self._to_jbook(source=source, dest=dest)
            else:
                self._to_jbook(source=source, dest=dest)

    def _user_approves(self, dest: str) -> bool:
        """Prompts the user to approve the conversion."""
        x = input(f"The file {dest} already exists. Type 'YES' to overwrite.")
        if x == "YES":
            return True
        else:
            return False

    def _to_notebook(self, source: str, dest: str) -> None:
        """Converts a single myst markdown file to jupyter notebook."""
        nb = jupytext.read(source)
        jupytext.write(nb, dest, fmt="ipynb")
        logger.info(f"Notebook converted from {source} to {dest}")

    def _to_jbook(self, source: str, dest: str) -> None:
        """Converts a single myst markdown file to jupyter notebook."""
        nb = jupytext.read(source)
        jupytext.write(nb, dest, fmt="md:myst")
        logger.info(f"JBook markdown converted from {source} to {dest}")
