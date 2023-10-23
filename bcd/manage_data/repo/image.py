#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/manage_data/repo/image.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 07:41:24 pm                                              #
# Modified   : Sunday October 22nd 2023 09:10:33 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import logging
from typing import Union, Callable

import pandas as pd
import pymysql
import numpy as np
from sqlalchemy.dialects.mssql import VARCHAR, DATETIME, INTEGER, FLOAT, TINYINT, BIGINT

from bcd.manage_data.repo.base import Repo
from bcd.manage_data.entity.image import Image, ImageFactory
from bcd.manage_data.database.base import Database
from bcd.manage_data.io.image import ImageIO

# ------------------------------------------------------------------------------------------------ #
IMAGE_DTYPES = {
    "id": VARCHAR(length=64),
    "case_id": VARCHAR(length=64),
    "mode": VARCHAR(length=8),
    "stage_id": TINYINT(),
    "stage": VARCHAR(length=64),
    "cancer": TINYINT(),
    "bit_depth": TINYINT(),
    "height": INTEGER(),
    "width": INTEGER(),
    "size": BIGINT(),
    "aspect_ratio": FLOAT(),
    "min_pixel_value": INTEGER(),
    "max_pixel_value": INTEGER(),
    "range_pixel_values": INTEGER(),
    "mean_pixel_value": FLOAT(),
    "median_pixel_value": INTEGER(),
    "std_pixel_value": FLOAT(),
    "filepath": VARCHAR(length=256),
    "fileset": VARCHAR(length=8),
    "created": DATETIME(),
    "task": VARCHAR(length=64),
    "taskrun_id": VARCHAR(length=64),
}


# ------------------------------------------------------------------------------------------------ #
class ImageRepo(Repo):
    __tablename = "image"

    def __init__(self, database: Database, image_factory: ImageFactory) -> None:
        super().__init__()
        self._database = database
        self._image_factory = image_factory
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def database(self) -> Database:
        return self._database

    def add(self, image: Image) -> None:
        """Adds an image to the repository

        Args:
            data (Union[Image, pd.DataFrame]): An Image object or a DataFrame
                containing image metadata.

        """
        if not self.exists(id=image.id):
            self._write_image(pixel_data=image.pixel_data, filepath=image.filepath)
            data = image.as_df()
            self._database.insert(
                data=data, tablename=self.__tablename, dtype=IMAGE_DTYPES, if_exists="append"
            )
        else:
            msg = f"Image {image.id} already exists."
            self._logger.exception(msg)
            raise FileExistsError(msg)

    def get(
        self, condition: Callable = None, metadata_only: bool = False
    ) -> Union[pd.DataFrame, tuple]:
        """Returns case images and their metadata matching the condition

        If a condition is not specified, metadata only will be returned.
        An example of a condition: condition = lambda df: df['stage_id'] > 0

        Args:
            condition (Callable): A lambda expression used to subset the data.
            metadata_only (bool): If True, only metadata are returned. If condition
                is None, only metadata are returned.  Default is False.

        Returns:
            If metadata_only is True, a DataFrame is returned. Otherwise, a tuple containing
                the metadata DataFrame and a dictionary of Image objects indexed by id is returned.
        """
        images = {}

        query = f"SELECT * FROM {self.__tablename};"
        params = None
        image_meta = self._database.query(query=query, params=params)

        if condition is None:
            return image_meta
        else:
            image_meta = image_meta[condition]
            if metadata_only:
                return image_meta
            else:
                for _, meta in image_meta.iterrows():
                    image = self._get(id=meta["id"])
                    images[meta["id"]] = image
                return (image_meta, images)

    def exists(self, id: str) -> bool:
        """Evaluates existence of an image by identifier.

        Args:
            id (str): Image UUID

        Returns:
            Boolean indicator of existence.
        """
        query = f"SELECT EXISTS(SELECT 1 FROM {self.__tablename} WHERE id = :id);"
        params = {"id": id}
        try:
            exists = self._database.exists(query=query, params=params)
        except pymysql.Error as e:
            self._logger.exception(e)
            raise
        else:
            return exists

    def count(self, condition: Callable = None) -> int:
        """Counts images matching the condition

        Args:
            condition (Callable): A lambda expression used to subset the data.

        Returns:
            Integer count of images matching condition.
        """
        query = f"SELECT * FROM {self.__tablename};"
        params = None
        try:
            image_meta = self._database.query(query=query, params=params)

        except pymysql.Error as e:
            self._logger.exception(e)
            raise
        else:
            if condition is not None:
                image_meta = image_meta[condition]
            return len(image_meta)

    def delete(self, condition: Callable) -> None:
        """Removes images matching the condition.

        Args:
            condition (Callable): Lambda expression subsetting the data.
        """
        query = f"SELECT * FROM {self.__tablename}"
        params = None
        image_meta = self._database.query(query=query, params=params)
        image_meta = image_meta[condition]

        for _, image in image_meta.iterrows():
            self._delete(id=image["id"])

    def _get(self, id: str) -> Image:
        """Obtains an image by identifier.

        Args:
            id (str): Image id

        Returns:
            Image object.
        """
        query = f"SELECT * FROM {self.__tablename} WHERE id = :id;"
        params = {"id": id}
        try:
            image_meta = self._database.query(query=query, params=params)
        except Exception as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            if len(image_meta) == 0:
                msg = f"Image id {id} does not exist."
                self._logger.exception(msg)
                raise FileNotFoundError(msg)
            return self._image_factory.from_df(df=image_meta)

    def _delete(self, id: str) -> None:
        """Removes an image and its metadata from the repository.

        Args:
            id (str): Image identifier
        """
        query = f"SELECT * FROM {self.__tablename} WHERE id = :id;"
        params = {"id": id}
        image_meta = self._database.query(query=query, params=params)

        try:
            os.remove(image_meta["filepath"].values[0])
        except OSError:
            msg = f"Image id: {id} does not exist at {image_meta['filepath'].values[0]}"
            self._logger.info(msg)
        finally:
            query = f"DELETE FROM {self.__tablename} WHERE id = :id;"
            params = {"id": id}
            self._database.delete(query=query, params=params)

    def _write_image(self, pixel_data: np.ndarray, filepath: str) -> None:
        io = ImageIO()
        io.write(pixel_data=pixel_data, filepath=filepath)
