#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/dal/image.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 11:47:17 am                                              #
# Modified   : Saturday December 30th 2023 05:07:54 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

import cv2
import dicomsdl as dicom
import numpy as np
import pandas as pd

from bcd import DataClass


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
#                                       IMAGE                                                      #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Image(DataClass):
    """Encapsulates an image object."""

    patient_id: str
    uid: str
    series_uid: str
    case_id: str
    series_description: str
    image_view: str
    breast_density: int
    subtlety: int
    breast_side: str
    abnormality_id: int
    abnormality_type: str
    calc_type: str
    calc_distribution: str
    mass_shape: str
    mass_margins: str
    height: int
    width: int
    size: int
    aspect_ratio: float
    bit_depth: int
    min_pixel_value: int
    max_pixel_value: int
    mean_pixel_value: float
    std_pixel_value: float
    pathology: str
    assessment: int
    cancer: bool
    fileset: str
    filepath: str
    pixels: np.ndarray

    @classmethod
    def create(cls, pixels: np.ndarray, meta: pd.Series) -> Image:
        return cls(
            patient_id=meta["patient_id"],
            uid=meta["uid"],
            series_uid=meta["series_uid"],
            case_id=meta["case_id"],
            series_description=meta["series_description"],
            image_view=meta["image_view"],
            breast_density=meta["breast_density"],
            subtlety=meta["subtlety"],
            breast_side=meta["breast_side"],
            abnormality_id=meta["abnormality_id"],
            abnormality_type=meta["abnormality_type"],
            calc_type=meta["calc_type"],
            calc_distribution=meta["calc_distribution"],
            mass_shape=meta["mass_shape"],
            mass_margins=meta["mass_margins"],
            height=meta["height"],
            width=meta["width"],
            size=meta["size"],
            aspect_ratio=meta["aspect_ratio"],
            bit_depth=meta["bit_depth"],
            min_pixel_value=meta["min_pixel_value"],
            max_pixel_value=meta["max_pixel_value"],
            mean_pixel_value=meta["mean_pixel_value"],
            std_pixel_value=meta["std_pixel_value"],
            pathology=meta["pathology"],
            assessment=meta["assessment"],
            cancer=meta["cancer"],
            fileset=meta["fileset"],
            filepath=meta["filepath"],
            pixels=pixels,
        )


# ------------------------------------------------------------------------------------------------ #
#                                      IMAGE IO                                                    #
# ------------------------------------------------------------------------------------------------ #
class ImageIO:
    """Manages IO of Image objects."""

    @classmethod
    def read(cls, filepath: str, channels: int = 2) -> np.ndarray:
        """Reads an image in DICOM, png, or jpeg format

        Args:
            filepath (str): Path to the file
            channels (int): Number of channels to read. This applies to non-DICOM
             images only.
        """
        filepath = cls._parse_filepath(filepath=filepath)
        if "dcm" in filepath:
            return cls._read_dicom(filepath=filepath)
        else:
            return cls._read_image(filepath=filepath, channels=channels)

    @classmethod
    def write(cls, pixel_data: np.ndarray, filepath: str, force: bool = False) -> None:
        filepath = cls._parse_filepath(filepath=filepath)
        if force or not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
    def _read_dicom(cls, filepath: str) -> np.ndarray:
        try:
            ds = dicom.open(filepath)
        except Exception as e:
            logging.exception(e)
            raise
        else:
            return ds.pixelData()

    @classmethod
    def _read_image(cls, filepath: str, channels: int = 2) -> np.ndarray:
        try:
            if channels == 2:
                return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            else:
                return cv2.imread(filepath)
        except FileNotFoundError as e:
            logging.exception(e)
            raise
        except Exception as e:
            logging.exception(e)
            raise

    @classmethod
    def _parse_filepath(cls, filepath: str) -> str:
        if isinstance(filepath, pd.Series):
            filepath = filepath.values[0]
        filepath = os.path.abspath(filepath)
        return filepath


# ------------------------------------------------------------------------------------------------ #
#                                        IMAGE READER                                              #
# ------------------------------------------------------------------------------------------------ #
class ImageReader:
    """Iterator class that serves up images matching user designated criteria.

    Args:
       case_filepath (str): Path to file containing cases and image paths.
       n (int): Number of images to include in the iteration.
       condition (Callable): A lambda statement to subset the data. For instance, obtaining
           all calcification case images would require the following lambda condition:
               lambda x: x['abnormality_type' == 'calc']

    """

    # TODO: Fix image reader if needed.
    def __init__(
        self, case_filepath: str, n: int = None, condition: Callable = None
    ) -> None:
        self._case_filepath = case_filepath
        self._n = n
        self._condition = condition
        self._idx = 0
        self._meta = None

    def __iter__(self) -> ImageReader:
        self._idx = 0
        df = pd.read_csv(self._case_filepath)
        if self._condition is not None:
            df = df.loc[self._condition]
        if self._n is not None:
            df = df.sample(n=self._n)
        self._meta = df
        return self

    def __next__(self) -> Image:
        try:
            meta = self._meta.iloc[self._idx]
            filepath = os.path.join()
            pixels = ImageIO.read(filepath=meta["filepath"])
            image = Image.create(pixels=pixels, meta=meta)
            self._idx += 1
            return image
        except IndexError as exc:
            raise StopIteration from exc
