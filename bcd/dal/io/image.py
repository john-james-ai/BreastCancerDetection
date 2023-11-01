#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/io/image.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 11:47:17 am                                              #
# Modified   : Wednesday November 1st 2023 03:42:59 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
import os
from typing import Union

import cv2
import numpy as np
import pandas as pd
import pydicom
from dependency_injector.wiring import Provide, inject

from bcd.config import Config
from bcd.container import BCDContainer
from bcd.core.image import Image
from bcd.dal.repo.image import ImageRepo


# ------------------------------------------------------------------------------------------------ #
class ImageIO:
    """Manages IO of Image objects."""

    @classmethod
    def read(cls, filepath) -> np.ndarray:
        filepath = cls._parse_filepath(filepath=filepath)
        if "dcm" in filepath:
            return cls._read_dicom(filepath=filepath)
        else:
            return cls._read_image(filepath=filepath)

    @classmethod
    def write(cls, pixel_data: np.ndarray, filepath: str, force: bool = False) -> None:
        filepath = cls._parse_filepath(filepath=filepath)
        if force or not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # pylint: disable=no-member
            cv2.imwrite(filepath, pixel_data)
        elif os.path.exists(filepath):
            msg = f"Image filename {os.path.realpath(filepath)} \
            already exists. If you wish to overwrite, set force = True"
            logging.warning(msg)

    @classmethod
    def delete(cls, filepath: str, silent: bool = False) -> None:
        """Deletes an image"""
        try:
            os.remove(filepath)
        except OSError:
            if not silent:
                msg = f"Delete warning! No image exists at {filepath}."
                logging.warning(msg)

    @classmethod
    def get_filepath(cls, uid: str, basedir: str, fileformat: str) -> str:
        filename = uid + "." + fileformat
        return os.path.join(basedir, filename)

    @classmethod
    def _read_dicom(cls, filepath: str) -> np.ndarray:
        try:
            ds = pydicom.dcmread(filepath)
        except Exception as e:
            logging.exception(e)
            raise
        else:
            return ds.pixel_array

    @classmethod
    def _read_image(cls, filepath: str) -> np.ndarray:
        try:
            # pylint: disable=no-member
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        except FileNotFoundError as e:
            logging.exception(e)
            raise
        except Exception as e:
            logging.exception(e)
            raise
        else:
            return image

    @classmethod
    def _parse_filepath(cls, filepath: str) -> str:
        if isinstance(filepath, pd.Series):
            filepath = filepath.values[0]
        filepath = os.path.abspath(filepath)
        return filepath


# ------------------------------------------------------------------------------------------------ #
#                                   IMAGE READER                                                   #
# ------------------------------------------------------------------------------------------------ #
class ImageReader:
    """Class that implements an image iterator.

    Args:
        n (int): Number of images to read. Cannot be used with frac.
        frac (float): Proportion of available images to read. Cannot be used with n.
        groupby (Union[str, list]): Grouping variables. If provided, stratified sampling will
            be conducted where n (or frac) images rom each group will be returned.
        image_repo (ImageRepo): Image repository
        config (Config): The config object which provides access to
    """

    @inject
    def __init__(
        self,
        n: int = None,
        frac: float = None,
        groupby: Union[str, list] = None,
        image_repo: ImageRepo = Provide[BCDContainer.dal.image_repo],
        config: Config = Config,
    ) -> None:
        self._n = n
        self._frac = frac
        self._groupby = groupby
        self._image_repo = image_repo
        self._config = config
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

        self._metadata = None
        self._max = None
        self._idx = None

    def __iter__(self) -> ImageReader:
        """Initializes the iterator."""
        self._idx = 0
        self._metadata = self._get_metadata()
        self._max = self._metadata.shape[0]
        return self

    def __next__(self) -> Image:
        """Returns the next image."""
        if self._idx < self._max:
            image_meta = self._metadata.iloc[self._idx]
            return self._image_repo.get(uid=image_meta["uid"].values[0])
        else:
            raise StopIteration

    def _get_metadata(self) -> pd.DataFrame:
        """Returns image metadata"""

        if self._n is not None and self._frac is not None:
            msg = "Cannot pass both n and frac. Either n, frac or both must be None"
            self._logger.exception(msg=msg)
            raise ValueError(msg)

        meta = self._image_repo.get_meta()
        return meta.groupby(by=self._groupby).sample(n=self._n, frac=self._frac)
