#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/core/image/repo.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 07:41:24 pm                                              #
# Modified   : Friday October 27th 2023 02:04:07 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import logging
from datetime import datetime
from typing import Union, Callable

import pandas as pd
import numpy as np
import pymysql
from sqlalchemy.dialects.mssql import VARCHAR, DATETIME, INTEGER, FLOAT, TINYINT, BIGINT

from bcd.config import Config
from bcd.core.base import Repo
from bcd.core.image.entity import Image
from bcd.core.image.factory import ImageFactory
from bcd.infrastructure.database.base import Database
from bcd.infrastructure.io.cache import ImageCache

# ------------------------------------------------------------------------------------------------ #
WRITE_IMAGE_DTYPES = {
    "uid": VARCHAR(length=64),
    "case_id": VARCHAR(length=64),
    "mode": VARCHAR(length=8),
    "stage_uid": INTEGER(),
    "stage": VARCHAR(length=64),
    "left_or_right_breast": VARCHAR(length=8),
    "image_view": VARCHAR(4),
    "abnormality_type": VARCHAR(24),
    "assessment": INTEGER(),
    "breast_density": INTEGER(),
    "bit_depth": INTEGER(),
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
    "preprocessor": VARCHAR(length=64),
    "cancer": TINYINT(),
    "task_id": VARCHAR(length=64),
    "created": DATETIME(),
}

# ------------------------------------------------------------------------------------------------ #
READ_IMAGE_DTYPES = {
    "uid": str,
    "case_id": str,
    "mode": str,
    "stage_uid": np.int8,
    "stage": str,
    "left_or_right_breast": str,
    "image_view": str,
    "abnormality_type": str,
    "assessment": np.int8,
    "breast_density": np.int8,
    "bit_depth": np.int8,
    "height": np.int64,
    "width": np.int64,
    "size": np.int64,
    "aspect_ratio": np.float64,
    "min_pixel_value": np.int64,
    "max_pixel_value": np.int64,
    "range_pixel_values": np.int64,
    "mean_pixel_value": np.float64,
    "median_pixel_value": np.int64,
    "std_pixel_value": np.float64,
    "filepath": str,
    "fileset": str,
    "preprocessor": str,
    "cancer": bool,
    "task_id": str,
}
PARSE_DATES = {'created': {'errors': 'ignore', 'yearfirst': True, 'infer_datetime_format': True}}
# ------------------------------------------------------------------------------------------------ #
class ImageRepo(Repo):
    """Image repository"""
    __tablename = "image"

    def __init__(
        self, database: Database, image_factory: ImageFactory, config: Config, cache: ImageCache
    ) -> None:
        super().__init__()
        self._database = database
        self._image_factory = image_factory
        self._config = config()
        self._autocommit = self._config.autocommit
        self._cache = cache()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def mode(self) -> str:
        """Returns the current mode."""
        return self._config.mode

    def add(self, image: Image) -> None:
        """Adds an image to the repository

        Args:
            image (Image): An Image object.

        """
        try:
            exists = self.exists(uid=image.uid)
        except Exception:  # pragma: no cover
            exists = False
        finally:
            if not exists:
                data = image.as_df()

                self._database.insert(
                    data=data, tablename=self.__tablename, dtype=WRITE_IMAGE_DTYPES, if_exists="append"
                )
                if self._autocommit:
                    self._cache.put(entity=image, write_through=True)
                else:
                    self._cache.put(image)
            else:
                msg = f"Image {image.uid} already exists."
                self._logger.exception(msg)
                raise FileExistsError(msg)

    def get(self, uid: str) -> Image:
        """Obtains an image by identifier.

        Args:
            id (str): Image id

        Returns:
            Image object.
        """
        query = f"SELECT * FROM {self.__tablename} WHERE uid = :uid;"
        params = {"uid": uid}
        try:
            image_meta = self._database.query(query=query,dtype=READ_IMAGE_DTYPES, params=params, parse_dates=PARSE_DATES)
        except Exception as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            if len(image_meta) == 0:  # pragma: no cover
                msg = f"Image id {uid} does not exist."
                self._logger.warning(msg)
                raise FileNotFoundError(msg)
            return self._image_factory.from_df(df=image_meta)

    def get_by_stage(
        self,
        stage_uid: int,
        n: int = None,
        frac: float = None,
        random_state: int = None,
    ) -> dict:
        """Returns images and metadata for the given stage.

        Args:
            stage_uid (int): The stage of the preprocessing cycle.
            n (int): Number of images to return. Cannot be used with frac.
            frac (float): Fraction of items matching condition to return. Cannot be used with n.
            random_stage (int): Seed for pseudo randomizing

        Returns:
            Tuple containing the metadata and dictionary of images, keyed by image id.
        """
        if n is not None and frac is not None:
            msg = "n and frac cannot be used together. Either n or frac must be None."
            self._logger.exception(msg)
            raise ValueError(msg)

        images = {}

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode AND stage_uid = :stage_uid;"
        params = {"mode": self.mode, "stage_uid": stage_uid}
        image_meta = self._database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist for Stage {stage_uid} in {self.mode} mode."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        if n is not None:
            image_meta = image_meta.groupby(by="preprocessor").sample(
                n=n, replace=False, random_state=random_state
            )
        elif frac is not None:
            image_meta = image_meta.groupby(by="preprocessor").sample(
                frac=frac, replace=False, random_state=random_state
            )

        for _, meta in image_meta.iterrows():
            image = self.get(uid=meta["uid"])
            images[meta["uid"]] = image
        return (
            image_meta,
            images,
        )

    def get_by_mode(
        self,
        n: int = None,
        frac: float = None,
        random_state: int = None,
    ) -> dict:
        """Returns images and metadata for the current mode.

        Samples are stratified by stage_uid and preprocessor. For instance, if
        n = 3, 3 images will be sampled from each stage_uid and preprocessor.

        Args:
            n (int): Number of images to return. Cannot be used with frac.
            frac (float): Fraction of items matching condition to return. Cannot be used with n.
            random_stage (int): Seed for pseudo randomizing

        Returns:
            Tuple containing the metadata and dictionary of images, keyed by image id.
        """
        if n is not None and frac is not None:
            msg = "n and frac cannot be used together. Either n or frac must be None."
            self._logger.exception(msg)
            raise ValueError(msg)

        images = {}

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        image_meta = self._database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist in {self.mode} mode."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        if n is not None:
            image_meta = image_meta.groupby(by=["stage_uid", "preprocessor"]).sample(
                n=n, replace=False, random_state=random_state
            )
        elif frac is not None:
            image_meta = image_meta.groupby(by=["stage_uid", "preprocessor"]).sample(
                frac=frac, replace=False, random_state=random_state
            )

        for _, meta in image_meta.iterrows():
            image = self.get(uid=meta["uid"])
            images[meta["uid"]] = image
        return (
            image_meta,
            images,
        )

    def get_by_preprocessor(
        self, preprocessor: str, n: int = None, frac: float = None, random_state: int = None
    ) -> list:
        """Returns a list of images for a given preprocessor.

        Args:
            preprocessor (str): The preprocessor that created the images
            n (int): Number of images to return. Cannot be used with frac.
            frac (float): Fraction of items matching condition to return. Cannot be used with n.
            random_stage (int): Seed for pseudo randomizing
        """
        if n is not None and frac is not None:
            msg = "n and frac cannot be used together. Either n or frac must be None."
            self._logger.exception(msg)
            raise ValueError(msg)

        images = {}

        query = (
            f"SELECT * FROM {self.__tablename} WHERE mode = :mode AND preprocessor = :preprocessor;"
        )
        params = {"mode": self.mode, "preprocessor": preprocessor}
        image_meta = self._database.query(query=query, params=params)

        if n is not None:
            image_meta = image_meta.sample(n=n, random_state=random_state)
        elif frac is not None:
            image_meta = image_meta.sample(frac=frac, random_state=random_state)

        if len(image_meta) == 0:
            msg = f"No images exist for the {preprocessor} preprocessor in {self.mode} mode."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        for _, meta in image_meta.iterrows():
            image = self.get(uid=meta["uid"])
            images[meta["uid"]] = image

        return (
            image_meta,
            images,
        )

    def get_meta(self, condition: Callable = None) -> Union[pd.DataFrame, list]:
        """Returns case images metadata
        Args:
            condition (Callable): A lambda expression used to subset the data.
                An example of a condition: condition = lambda df: df['stage_uid'] > 0

        """
        query = f"SELECT * FROM {self.__tablename};"
        params = None
        image_meta = self._database.query(query=query, params=params)

        if condition is None:
            return image_meta
        else:
            return image_meta[condition]

    def exists(self, uid: str) -> bool:
        """Evaluates existence of an image by identifier.

        Args:
            id (str): Image UUID

        Returns:
            Boolean indicator of existence.
        """
        query = f"SELECT EXISTS(SELECT 1 FROM {self.__tablename} WHERE uid = :uid);"
        params = {"uid": uid}
        try:
            exists = self._database.exists(query=query, params=params)
        except pymysql.Error as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            return exists

    def count(self, condition: Callable = None) -> int:
        """Counts images matching the condition for the current mode.

        Args:
            condition (Callable): A lambda expression used to subset the data.

        Returns:
            Integer count of images matching condition.
        """
        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        try:
            image_meta = self._database.query(query=query, params=params)

        except pymysql.Error as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            if condition is not None:
                image_meta = image_meta[condition]
            return len(image_meta)

    def delete(self, uid: str, filepath: str) -> None:
        """Removes an image and its metadata from the repository.

        Args:
            uid (str): The unique identifier for the image
            filepath (str): The path to the images on disk.
        """

        try:
            os.remove(filepath)
        except OSError:  # pragma: no cover
            msg = f"Image id: {uid} does not exist at {filepath}"
            self._logger.warning(msg)
        finally:
            query = f"DELETE FROM {self.__tablename} WHERE uid = :uid;"
            params = {"uid": uid}
            self._database.delete(query=query, params=params)
            self._cache.remove(uid=uid)

    def delete_by_stage(self, stage_uid: int) -> None:
        """Removes images for a given stage

        Args:
            stage_uid (int): The stage
        """
        if self._delete_permitted(stage_uid=stage_uid):
            query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode and stage_uid = :stage_uid;"
            params = {"mode": self.mode, "stage_uid": stage_uid}
            image_meta = self._database.query(query=query, params=params)

            if len(image_meta) == 0:
                msg = f"No images exist for stage {stage_uid} in {self.mode} mode."
                self._logger.warning(msg)

            for _, image in image_meta.iterrows():
                self.delete(uid=image["uid"], filepath=image['filepath'])

        else:
            msg = f"Delete of stage {stage_uid} images not permitted without confirmation."
            self._logger.info(msg)

    def delete_by_preprocessor(self, preprocessor: str) -> None:
        """Removes images for a given preprocessor.

        Args:
            preprocessor (str): The preprocessor that created the image.
        """

        query = (
            f"SELECT * FROM {self.__tablename} WHERE mode = :mode and preprocessor = :preprocessor;"
        )
        params = {"mode": self.mode, "preprocessor": preprocessor}
        image_meta = self._database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist for {preprocessor} preprocessor in {self.mode} mode."
            self._logger.warning(msg)

        for _, image in image_meta.iterrows():
            self.delete(uid=image["uid"], filepath=image["filepath"])


    def delete_by_mode(self) -> None:
        """Removes images for a current mode."""

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        image_meta = self._database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist in {self.mode} mode."
            self._logger.warning(msg)

        for _, image in image_meta.iterrows():
            self.delete(uid=image["uid"], filepath=image["filepath"])


    def begin(self) -> None:
        """Begins a transaction."""
        self._cache.reset()

    def save(self) -> None:
        """Commits changes to the database, and saves cache to file."""
        self._cache.save()

    def rollback(self) -> None:
        """Commits changes to the database, and saves cache to file."""
        self._cache.reset()

    def _delete_permitted(self, stage_uid: int) -> bool:
        if stage_uid == 0:
            go = input("Please confirm that you wish to delete Stage 0 images [Y/N]")
            if "y" in go.lower():
                return True
            else:
                return False
        else:
            return True
