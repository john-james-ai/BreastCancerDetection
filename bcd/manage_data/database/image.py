#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/manage_data/database/image.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 07:41:24 pm                                              #
# Modified   : Sunday October 22nd 2023 03:58:14 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import logging
from typing import Union, Callable

import pandas as pd
import numpy as np
import cv2
import pymysql

from bcd.manage_data.entity.image import Image
from bcd.manage_data.database.base import Database


# ------------------------------------------------------------------------------------------------ #
class ImageRepo:
    __tablename = "image"

    def __init__(self, database: Database) -> None:
        self._database = database
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def add(self, image: Image) -> None:
        """Adds an image to the repository

        Args:
            data (Union[Image, pd.DataFrame]): An Image object or a DataFrame
                containing image metadata.

        """
        condition = lambda df: df["id"] == image.id  # noqa
        if not self.exists(condition=condition):
            self._write_image(pixel_data=image.pixel_data, filepath=image.filepath)
            data = image.as_df()
            self._database.insert(data=data, tablename=self.__tablename, if_exists="append")
        else:
            msg = f"Image {image.id} already exists."
            self._logger.exception(msg)
            raise FileExistsError(msg)

    def exists(self, condition: Callable) -> bool:
        """Returns True if one or more images matching the condition exist(s), False otherwise.

        Args:
            condition (Callable): A lambda expression used to subset the data.

        Returns:
            Boolean indicator of existence.
        """
        query = f"SELECT * FROM {self.__tablename};"
        params = None
        try:
            image_meta = self._database.query(query=query, params=params)
            image_meta = image_meta[condition]
        except Exception:
            return False
        else:
            return len(image_meta) > 0

    def get_image(self, id: str) -> Image:
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
        except pymysql.err.ProgrammingError as e:
            self._logger.exception(e)
            raise
        else:
            if len(image_meta) == 0:
                msg = f"Image id {id} does not exist."
                self._logger.exception(msg)
                raise FileNotFoundError(msg)
            return Image.from_df(df=image_meta)

    def get_images(
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
                    image = self.get_image(id=meta["id"])
                    images[meta["id"]] = image
                return (image_meta, images)

    def delete_image(self, id: str, force: bool = False) -> None:
        """Removes an image and its metadata from the repository.

        Args:
            id (str): Image identifier
            force (bool): If True, stage 0, original images will be deleted.
        """
        query = f"SELECT * FROM {self.__tablename} WHERE id = :id;"
        params = {"id": id}
        image_meta = self._database.query(query=query, params=params)

        if not force and image_meta["stage_id"].values[0] == 0:
            msg = f"Image {id} is a stage 0, original image. Original images cannot be deleted."
            self._logger.info(msg)

        else:
            try:
                os.remove(image_meta["filepath"].values[0])
            except OSError:
                msg = f"Image id: {id} does not exist at {image_meta['filepath'].values[0]}"
                self._logger.info(msg)

            finally:
                query = f"DELETE FROM {self.__tablename} WHERE id = :id;"
                params = {"id": id}
                self._database.delete(query=query, params=params)

    def delete_images(self, condition: Callable, force: bool = False) -> None:
        """Removes images matching the condition.

        Args:
            condition (Callable): Lambda expression subsetting the data.
            force (bool): If True, stage 0, original images will be deleted.
        """
        query = f"SELECT * FROM {self.__tablename}"
        params = None
        image_meta = self._database.query(query=query, params=params)
        image_meta = image_meta[condition]

        for _, image in image_meta.iterrows():
            self.delete_image(id=image["id"], force=force)

    def _read_image(self, filepath: str) -> np.ndarray:
        abs_filepath = os.path.abspath(filepath)
        try:
            image = cv2.imread(abs_filepath)
        except FileNotFoundError:
            msg = f"No image found at {filepath}"
            self._logger.exception(msg)
            raise
        else:
            return image

    def _write_image(self, pixel_data: np.ndarray, filepath: str) -> None:
        abs_filepath = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_filepath), exist_ok=True)
        cv2.imwrite(filename=abs_filepath, img=pixel_data)
