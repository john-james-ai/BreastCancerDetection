#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/prep/image/io.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 17th 2023 04:48:30 pm                                               #
# Modified   : Wednesday October 18th 2023 06:23:06 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from glob import glob

import pandas as pd
import numpy as np
import pydicom
import cv2

# ------------------------------------------------------------------------------------------------ #
DICOM_META_FILEPATH = "data/clean/dicom_case.csv"


# ------------------------------------------------------------------------------------------------ #
class DICOMIO:
    def __init__(self) -> None:
        dicom_meta_filepath = os.path.abspath(DICOM_META_FILEPATH)
        self._dicom_meta = pd.read_csv(dicom_meta_filepath)

    def get_paths(self, n: int = None, directory: str = None, fileset: str = "train") -> list:
        """Returns a list of file paths to images.

        Args:
            n (int): Number of paths to return. If None, all paths matching criteria will be returned.
            directory (str): The folder within the data directory.
            fileset (str): Either 'train', or 'test'. Default = 'train'

        """
        if directory is None or "raw" in directory:
            filepaths = list(
                self._dicom_meta.loc[
                    (self._dicom_meta["series_description"] == "full mammogram images")
                    & (self._dicom_meta["fileset"] == fileset)
                ]["filepath"]
            )

        else:
            pattern_png = directory + "/**/*.png"
            pattern_dcm = directory + "/**/*.dcm"
            filepaths = glob(pattern_png, recursive=True)
            filepaths.extend(glob(pattern_dcm, recursive=True))
            filepaths = [filepath for filepath in filepaths if fileset in filepath.lower()]

        if n is not None:
            n = min(n, len(filepaths))
            filepaths = np.random.choice(filepaths, size=n, replace=False)

        return filepaths

    def get_path(
        self, source_filepath: str, source_dir: str, destination_dir: str, format: str = "png"
    ) -> str:
        """Returns a filepath in the designated directory for png or jpeg format.

        Args:
            source_filepath (str): The filepath of the source image
            source_dir (str): The directory for the source image
            destination_dir (str): The destination directory for the path.
            format (str): Image file format.
        """
        dest_filepath = source_filepath.replace(source_dir, destination_dir)
        return dest_filepath.replace("dcm", format)

    def read_image(self, filepath: str) -> np.ndarray:
        """Reads an image from the designated filepath

        Args:
            filepath (str): The image filepath.
        """
        filepath = os.path.abspath(filepath)
        if "dcm" in os.path.splitext(filepath)[1]:
            return self._read_dicom_image(filepath=filepath)
        else:
            return self._read_image(filepath=filepath)

    def save_image(self, img: np.array, filepath: str) -> None:
        """Saves an image to the designated filepath

        Args:
            img (np.array): Image in a 2D numpy array
            filepath (str): Path to which the image will be saved.
        """
        filepath = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filename=filepath, img=img)

    def _read_dicom_image(self, filepath: str) -> np.ndarray:
        """Reads a DICOM image and returns the data as a numpy array"""
        dataset = pydicom.dcmread(filepath)
        return dataset.pixel_array

    def _read_image(self, filepath: str) -> np.ndarray:
        """Reads png, or jpeg image based upon filepath"""
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
