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
# Modified   : Thursday October 26th 2023 01:12:59 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import logging
from typing import Union, Callable

import pandas as pd
import pymysql
from sqlalchemy.dialects.mssql import VARCHAR, DATETIME, INTEGER, FLOAT, TINYINT, BIGINT

from bcd.config import Config
from bcd.core.base import Repo
from bcd.core.image.entity import Image
from bcd.core.image.factory import ImageFactory
from bcd.infrastructure.database.base import Database

# ------------------------------------------------------------------------------------------------ #
IMAGE_DTYPES = {
    "id": VARCHAR(length=64),
    "case_id": VARCHAR(length=64),
    "mode": VARCHAR(length=8),
    "stage_id": INTEGER(),
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
    "cancer": TINYINT(),
    "preprocessor": VARCHAR(length=64),
    "task_id": VARCHAR(length=64),
    "created": DATETIME(),
}


# ------------------------------------------------------------------------------------------------ #
class ImageRepo(Repo):
    __tablename = "image"

    def __init__(self, database: Database, image_factory: ImageFactory, config: Config) -> None:
        super().__init__()
        self._database = database
        self._image_factory = image_factory
        self._config = config()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def mode(self) -> str:
        return self._config.get_mode()

    def add(self, image: Image) -> None:
        """Adds an image to the repository

        Args:
            image (Image): An Image object.

        """
        try:
            exists = self.exists(id=image.id)
        except Exception:  # pragma: no cover
            exists = False
        finally:
            if not exists:
                data = image.as_df()

                self._database.insert(
                    data=data, tablename=self.__tablename, dtype=IMAGE_DTYPES, if_exists="append"
                )
            else:
                msg = f"Image {image.id} already exists."
                self._logger.exception(msg)
                raise FileExistsError(msg)

    def get(self, id: str) -> Image:
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
            if len(image_meta) == 0:  # pragma: no cover
                msg = f"Image id {id} does not exist."
                self._logger.exception(msg)
                raise FileNotFoundError(msg)
            return self._image_factory.from_df(df=image_meta)

    def get_by_stage(
        self,
        stage_id: int,
        n: int = None,
        frac: float = None,
        random_state: int = None,
    ) -> dict:
        """Returns images and metadata for the given stage.

        Args:
            stage_id (int): The stage of the preprocessing cycle.
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

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode AND stage_id = :stage_id;"
        params = {"mode": self.mode, "stage_id": stage_id}
        image_meta = self._database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist for Stage {stage_id} in {self.mode} mode."
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
            image = self.get(id=meta["id"])
            images[meta["id"]] = image
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

        Samples are stratified by stage_id and preprocessor. For instance, if
        n = 3, 3 images will be sampled from each stage_id and preprocessor.

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
            image_meta = image_meta.groupby(by=["stage_id", "preprocessor"]).sample(
                n=n, replace=False, random_state=random_state
            )
        elif frac is not None:
            image_meta = image_meta.groupby(by=["stage_id", "preprocessor"]).sample(
                frac=frac, replace=False, random_state=random_state
            )

        for _, meta in image_meta.iterrows():
            image = self.get(id=meta["id"])
            images[meta["id"]] = image
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
            image = self.get(id=meta["id"])
            images[meta["id"]] = image

        return (
            image_meta,
            images,
        )

    def get_meta(self, condition: Callable = None) -> Union[pd.DataFrame, list]:
        """Returns case images metadata
        Args:
            condition (Callable): A lambda expression used to subset the data.
                An example of a condition: condition = lambda df: df['stage_id'] > 0

        """
        query = f"SELECT * FROM {self.__tablename};"
        params = None
        meta = self._database.query(query=query, params=params)
        if condition is None:
            return meta
        else:
            return meta[condition]

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

    def delete(self, id: str, filepath: str) -> None:
        """Removes an image and its metadata from the repository."""
        try:
            os.remove(filepath)
        except OSError:  # pragma: no cover
            msg = f"Image id: {id} does not exist at {filepath}"
            self._logger.warn(msg)
        finally:
            query = f"DELETE FROM {self.__tablename} WHERE id = :id;"
            params = {"id": id}
            self._database.delete(query=query, params=params)

    def delete_by_stage(self, stage_id: int) -> None:
        """Removes images for a given stage

        Args:
            stage_id (int): The stage
        """
        if self._delete_permitted(stage_id=stage_id):
            query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode and stage_id = :stage_id;"
            params = {"mode": self.mode, "stage_id": stage_id}
            image_meta = self._database.query(query=query, params=params)

            if len(image_meta) == 0:
                msg = f"No images exist for stage {stage_id} in {self.mode} mode."
                self._logger.exception(msg)
                raise FileNotFoundError(msg)

            for _, image in image_meta.iterrows():
                self.delete(id=image["id"], filepath=image["filepath"])
        else:
            msg = f"Delete of stage {stage_id} images not permitted without confirmation."
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
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        for _, image in image_meta.iterrows():
            self.delete(id=image["id"], filepath=image["filepath"])

    def delete_by_mode(self) -> None:
        """Removes images for a current mode."""

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        image_meta = self._database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist in {self.mode} mode."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        for _, image in image_meta.iterrows():
            self.delete(id=image["id"], filepath=image["filepath"])

    def _delete_permitted(self, stage_id: int) -> bool:
        if stage_id == 0:
            go = input("Please confirm that you wish to delete Stage 0 images [Y/N]")
            if "y" in go.lower():
                return True
            else:
                return False
        else:
            return True
