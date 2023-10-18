#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/prep/image/base.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 16th 2023 08:41:25 pm                                                #
# Modified   : Wednesday October 18th 2023 07:16:21 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Preparation Base Module"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from bcd.data.prep.image.io import DICOMIO


# ------------------------------------------------------------------------------------------------ #
class DICOMTask(ABC):
    """Defines interface for DICOM preprocessing tasks."""

    def __init__(self, io: DICOMIO = DICOMIO()) -> None:
        self._io = io

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Executes the task on the input and returns the task output."""

    def get_paths(self, n: int = None, fileset: str = "train") -> list:
        """Returns a list of file paths to images.

        Args:
            n (int): Number of paths to return. If None, all paths matching criteria will be returned.
            fileset (str): Either 'train', or 'test'. Default = 'train'
            view (str): Image view. If None, paths for both CC and MLO views will be returned.

        """
        return self._io.get_paths(n=n, fileset=fileset)

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
        return self._io.get_path(
            source_filepath=source_filepath,
            source_dir=source_dir,
            destination_dir=destination_dir,
            format=format,
        )

    def read_image(self, filepath: str) -> np.array:
        """Reads an image from the designated filepath

        Args:
            filepath (str): The image filepath.
        """
        return self._io.read_image(filepath=filepath)

    def save_image(self, img: np.array, filepath: str) -> None:
        """Saves an image to the designated filepath

        Args:
            img (np.array): Image in a 2D numpy array
            filepath (str): Path to which the image will be saved.
        """
        self._io.save_image(img=img, filepath=filepath)

    def display(self, img: np.array, cmap: str = "jet", cbar: bool = True) -> None:
        """Displays a single mammogram image

        Args:
            img (np.array): Image pixel data in numpy array format.
            cmap (str): Color map. Default is 'jet'
            cbar (bool): Indicates whether a color bar is to be displayed.
        """
        plt.imshow(img, cmap=cmap)
        if cbar:
            plt.colorbar()
        plt.show()

    def visualize(self, directory: str, n: int = 8) -> None:
        """Visualize samples of images from the specified directory

        Args:
            directory (str): The directory containing the images
            n (int): The number of images to visualize
        """
        filepaths = self._io.get_paths(n=n, directory=directory)
        if len(filepaths) == 0:
            return

        n = min(n, len(filepaths))
        n = n if n % 2 == 0 else n - 1

        ncols = 4
        nrows = n // 4

        idx = 1
        fig = plt.figure(figsize=(ncols * 3, nrows * 3))
        for filepath in tqdm(filepaths, total=len(filepaths)):
            img = self.read_image(filepath=filepath)
            fig.add_subplot(nrows, ncols, idx)
            idx += 1
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            plt.title(filepath.split("/")[:-1][3])
        plt.show()
