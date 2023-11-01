#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/repo/image.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 07:41:24 pm                                              #
# Modified   : Wednesday November 1st 2023 01:51:02 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.dialects.mssql import BIGINT, DATETIME, FLOAT, INTEGER, TINYINT, VARCHAR

from bcd.config import Config
from bcd.dal.database.base import Database
from bcd.dal.io.image import ImageIO
from bcd.dal.repo.base import Repo
from bcd.preprocess.image.image import Image, ImageFactory

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-renamed, arguments-differ, broad-exception-caught
# ------------------------------------------------------------------------------------------------ #
WRITE_IMAGE_DTYPES = {
    "uid": VARCHAR(length=64),
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
    "method": VARCHAR(length=64),
    "cancer": TINYINT(),
    "task_id": VARCHAR(length=64),
    "created": DATETIME(),
}
# ------------------------------------------------------------------------------------------------ #
READ_IMAGE_DTYPES = {
    "uid": str,
    "case_id": str,
    "mode": str,
    "stage_id": np.int8,
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
    "method": str,
    "cancer": bool,
    "task_id": str,
}
PARSE_DATES = {"created": {"errors": "ignore", "yearfirst": True}}


# ------------------------------------------------------------------------------------------------ #
#                                     IMAGE REPO                                                   #
# ------------------------------------------------------------------------------------------------ #
class ImageRepo(Repo):
    """Image repository"""

    __tablename = "image"

    def __init__(
        self,
        database: Database,
        image_factory: type[ImageFactory] = ImageFactory,
        io: ImageIO = ImageIO,
        config: Config = Config,
    ) -> None:
        super().__init__(database=database)
        self._image_factory = image_factory
        self._io = io
        self._mode = config.get_mode()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def mode(self) -> str:
        """Returns the current mode."""
        return self._mode

    def add(self, image: Image) -> None:
        """Adds an image to the repository

        Args:
            image (Image): An Image object.

        """
        exists = False
        try:
            exists = self.exists(uid=image.uid)
        except Exception:  # pragma: no cover
            pass
        finally:
            if not exists:
                data = image.as_df()

                self.database.insert(
                    data=data,
                    tablename=self.__tablename,
                    dtype=WRITE_IMAGE_DTYPES,
                    if_exists="append",
                )
                self._io.write(pixel_data=image.pixel_data, filepath=image.filepath)
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
            image_meta = self.database.query(
                query=query, dtype=READ_IMAGE_DTYPES, params=params, parse_dates=PARSE_DATES
            )
        except sqlalchemy.exc.ProgrammingError as e:  # pragma: no cover
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
        stage_id: int,
    ) -> Tuple[pd.DataFrame, dict]:
        """Returns images and metadata for the given stage.

        Args:
            stage_id (int): The stage of the preprocessing cycle.
            random_stage (int): Seed for pseudo randomizing

        Returns:
            Tuple containing the metadata and dictionary of images, keyed by image id.
        """

        images = {}

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode AND stage_id = :stage_id;"
        params = {"mode": self.mode, "stage_id": stage_id}
        image_meta = self.database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist for Stage {stage_id} in {self.mode} mode."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        for _, meta in image_meta.iterrows():
            image = self.get(uid=meta["uid"])
            images[meta["uid"]] = image
        return (
            image_meta,
            images,
        )

    def get_by_mode(
        self,
    ) -> Tuple[pd.DataFrame, dict]:
        """Returns images and metadata for the current mode.

        Args:
            random_stage (int): Seed for pseudo randomizing

        Returns:
            Tuple containing the metadata and dictionary of images, keyed by image id.
        """

        images = {}

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        image_meta = self.database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist in {self.mode} mode."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        for _, meta in image_meta.iterrows():
            image = self.get(uid=meta["uid"])
            images[meta["uid"]] = image
        return (
            image_meta,
            images,
        )

    def get_by_method(self, method: str) -> Tuple[pd.DataFrame, dict]:
        """Returns a list of images for a given method.

        Args:
            method (str): The method that created the images
        Returns:
            Tuple containing the metadata and dictionary of images, keyed by image id.
        """
        images = {}

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode AND method = :method;"
        params = {"mode": self.mode, "method": method}
        image_meta = self.database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist for the {method} method in {self.mode} mode."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        for _, meta in image_meta.iterrows():
            image = self.get(uid=meta["uid"])
            images[meta["uid"]] = image

        return (
            image_meta,
            images,
        )

    def get_meta(self, condition: Callable = None) -> pd.DataFrame:
        """Returns case images metadata
        Args:
            condition (Callable): A lambda expression used to subset the data.
                An example of a condition: condition = lambda df: df['stage_id'] > 0

        """
        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        image_meta = self.database.query(query=query, params=params)

        if condition is None:
            return image_meta
        else:
            return image_meta[condition]

    def sample(
        self, n: int = None, frac: float = None, groupby: Union[str, list] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """Returns a sample of the images.

        Returns a sample of the images by count (n) or by proportion (frac).
        Samples can also be obtained using a grouping variables; whereby, each
        group is sampled according to n or frac.

        Args:
            n (int): Number of images to return. If groupby is not None, this is the
                number of images to be returned by group. Cannot be used with frac.
            frac (float): Fraction of existing images to return. If groupby is not None, this is
                the fraction of images to be returned by group. Cannot be used with n.
            groupby (Union[str,list]): Grouping variables.

        Raises:
            ValueError if n and frac are both not None

        Returns:
        """
        images = {}
        if n is not None and frac is not None:
            msg = "Either n or frac must be None."
            self._logger.exception(msg)
            raise ValueError(msg)

        if n is None and frac is None:
            msg = "n or frac must be provided."
            self._logger.exception(msg)
            raise ValueError(msg)

        # Extract all metadata
        image_meta = self.get_meta()

        # Sample from the metadata.
        if n is not None:
            if groupby is None:
                image_meta = image_meta.sample(n=n, replace=False)
            else:
                image_meta = image_meta.groupby(by=groupby).sample(n=n, replace=False)
        else:
            if groupby is None:
                image_meta = image_meta.sample(frac=frac, replace=False)
            else:
                image_meta = image_meta.groupby(by=groupby).sample(frac=frac, replace=False)

        # Obtain the images
        for _, meta in image_meta.iterrows():
            images[meta["uid"]] = self.get(uid=meta["uid"])

        return (image_meta, images)

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
            exists = self.database.exists(query=query, params=params)
        except Exception as e:  # pragma: no cover
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
            image_meta = self.database.query(query=query, params=params)

        except Exception as e:  # pragma: no cover
            self._logger.exception(e)
            raise
        else:
            if condition is not None:
                image_meta = image_meta[condition]
            return len(image_meta)

    def delete(self, uid: str, filepath: str, silent: bool = True) -> None:
        """Removes an image and its metadata from the repository.

        Args:
            uid (str): The unique identifier for the image
            filepath (str): The path to the images on disk.
            silent (bool): Whether to silently delete, even if file does
                not exist. False will log a warning.
        """

        query = f"DELETE FROM {self.__tablename} WHERE uid = :uid;"
        params = {"uid": uid}
        try:
            self.database.delete(query=query, params=params)
        except Exception as e:
            self._logger.exception(e)
            raise
        else:
            self._io.delete(filepath=filepath, silent=silent)

    def delete_by_stage(self, stage_id: int) -> None:
        """Removes images for a given stage

        Args:
            stage_id (int): The stage
        """
        if self._delete_permitted(stage_id=stage_id):
            query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode and stage_id = :stage_id;"
            params = {"mode": self.mode, "stage_id": stage_id}
            image_meta = self.database.query(query=query, params=params)

            if len(image_meta) == 0:
                msg = f"No images exist for stage {stage_id} in {self.mode} mode."
                self._logger.warning(msg)

            for _, image in image_meta.iterrows():
                self.delete(uid=image["uid"], filepath=image["filepath"])

        else:
            msg = f"Delete of stage {stage_id} images not permitted without confirmation."
            self._logger.info(msg)

    def delete_by_method(self, method: str) -> None:
        """Removes images for a given method.

        Args:
            method (str): The method that created the image.
        """

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode and method = :method;"
        params = {"mode": self.mode, "method": method}
        image_meta = self.database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist for {method} method in {self.mode} mode."
            self._logger.warning(msg)

        for _, image in image_meta.iterrows():
            self.delete(uid=image["uid"], filepath=image["filepath"])

    def delete_by_mode(self) -> None:
        """Removes images for a current mode."""

        query = f"SELECT * FROM {self.__tablename} WHERE mode = :mode;"
        params = {"mode": self.mode}
        image_meta = self.database.query(query=query, params=params)

        if len(image_meta) == 0:
            msg = f"No images exist in {self.mode} mode."
            self._logger.warning(msg)

        for _, image in image_meta.iterrows():
            self.delete(uid=image["uid"], filepath=image["filepath"])

    def _delete_permitted(self, stage_id: int) -> bool:
        if stage_id == 0:
            go = input("Please confirm that you wish to delete Stage 0 images [Y/N]")
            if "y" in go.lower():
                return True
            else:
                return False
        else:
            return True
