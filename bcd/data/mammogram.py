#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/explore/image/mammogram.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday December 19th 2023 04:10:38 pm                                              #
# Modified   : Thursday December 28th 2023 09:14:52 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bcd import DataClass
from bcd.dal.image import ImageIO
from bcd.utils.image import grayscale


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
#                                       IMAGE CLASS                                                #
# ------------------------------------------------------------------------------------------------ #
@dataclass(eq=False)
class Mammogram(DataClass):
    """Image Object"""

    uid: str
    patient_id: str
    case_id: str
    series_uid: str
    series_description: str
    left_or_right_breast: str
    image_view: str
    abnormality_type: str
    abnormality_id: int
    assessment: int
    breast_density: int
    subtlety: int
    bit_depth: int
    pixel_data: np.ndarray
    height: int
    width: int
    size: int
    aspect_ratio: float
    min_pixel_value: int
    max_pixel_value: int
    range_pixel_values: int
    mean_pixel_value: float
    median_pixel_value: int
    std_pixel_value: float
    filepath: str
    fileset: str
    cancer: bool

    def __eq__(self, other: Mammogram) -> bool:
        return (
            self.uid == other.uid
            and self.patient_id == other.patient_id
            and self.case_id == other.case_id
            and self.series_uid == other.series_uid
            and self.series_description == other.series_description
            and self.left_or_right_breast == other.left_or_right_breast
            and self.image_view == other.image_view
            and self.abnormality_type == other.abnormality_type
            and self.abnormality_id == other.abnormality_id
            and self.assessment == other.assessment
            and self.breast_density == other.breast_density
            and self.subtlety == other.subtlety
            and self.bit_depth == other.bit_depth
            and (self.pixel_data == other.pixel_data).all()
            and self.height == other.height
            and self.width == other.width
            and self.size == other.size
            and round(self.aspect_ratio, 1) == round(other.aspect_ratio, 1)
            and self.min_pixel_value == other.min_pixel_value
            and self.max_pixel_value == other.max_pixel_value
            and self.range_pixel_values == other.range_pixel_values
            and round(self.mean_pixel_value, 0) == round(other.mean_pixel_value, 0)
            and self.median_pixel_value == other.median_pixel_value
            and round(self.std_pixel_value, 1) == round(other.std_pixel_value, 1)
            and self.filepath == other.filepath
            and self.fileset == other.fileset
            and self.cancer == other.cancer
        )

    @classmethod
    def from_dict(cls, data: dict) -> Mammogram:
        """Creates a Mammogram object from a series obtained from DICOM image metadata.

        Args:
            data (dict): Series containing image metadata.
        """
        pixel_data = ImageIO.read(filepath=data["filepath"])
        return cls(
            uid=data["uid"],
            patient_id=data["patient_id"],
            case_id=data["case_id"],
            series_uid=data["series_uid"],
            series_description=data["series_description"],
            left_or_right_breast=data["left_or_right_breast"],
            image_view=data["image_view"],
            abnormality_type=data["abnormality_type"],
            abnormality_id=data["abnormality_id"],
            assessment=data["assessment"],
            breast_density=data["breast_density"],
            subtlety=data["subtlety"],
            bit_depth=data["bit_depth"],
            pixel_data=pixel_data,
            height=data["height"],
            width=data["width"],
            size=data["size"],
            aspect_ratio=data["aspect_ratio"],
            min_pixel_value=data["min_pixel_value"],
            max_pixel_value=data["max_pixel_value"],
            range_pixel_values=data["range_pixel_values"],
            mean_pixel_value=data["mean_pixel_value"],
            median_pixel_value=data["median_pixel_value"],
            std_pixel_value=data["std_pixel_value"],
            filepath=data["filepath"],
            fileset=data["fileset"],
            cancer=data["cancer"],
        )

    def difference(self, other: Mammogram) -> dict:
        dict1 = self.as_dict()
        dict2 = other.as_dict()
        set1 = set(dict1.items())
        set2 = set(dict2.items())
        return set1 ^ set2

    def visualize(
        self,
        cmap: str = None,
        ax: plt.Axes = None,
        figsize: tuple = (8, 8),
        actual_size: bool = True,
    ) -> plt.Axes:  # pragma: no cover
        """Plots the image on an axis

        Args:
            cmap (str): The colormap used to render the plot
            ax (plt.Axes): Matplotlib Axes object. Optional.
            figsize (tuple): Size of the image if a plt.Axes object is not provided.
                Default = (8,8)
            actual_size (bool): If True, the image is rendered at actual size
        """
        if actual_size:
            ax = self._visualize_actual_size(cmap)
        else:
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
            ax.imshow(self.pixel_data, cmap=cmap, aspect="auto")
            ax.axis("off")
        return ax

    def _visualize_actual_size(self, cmap: str) -> plt.Axes:  # pragma: no cover
        """Renders image at actual size"""
        dpi = 80
        height, width = self.pixel_data.shape

        # Computes the figure size to render the plot at actual size.
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Display the image.
        ax.imshow(self.pixel_data, cmap=cmap)

        # Hide spines, ticks, etc.
        ax.axis("off")

        return ax

    def histogram(self, ax: plt.Axes = None, figsize: tuple = (12, 4)) -> None:
        """Plots a histogram of image pixel values.

        Args:
            ax (plt.Axes): Matplotlib Axes object. Optional.
            figsize (tuple): Size of the image if a plt.Axes object is not provided.
                Default = (8,8)

        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        img = grayscale(self.pixel_data)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        _ = ax.plot(hist)

        _ = ax.set_xlabel("Pixel Values")

        title = f"Case {self.case_id}"
        _ = ax.set_title(title, fontsize=12)

        return ax

    def as_df(self) -> pd.DataFrame:
        d = {
            "uid": self.uid,
            "patient_id": self.patient_id,
            "case_id": self.case_id,
            "series_uid": self.series_uid,
            "series_description": self.series_description,
            "left_or_right_breast": self.left_or_right_breast,
            "image_view": self.image_view,
            "abnormality_type": self.abnormality_type,
            "abnormality_id": self.abnormality_id,
            "assessment": self.assessment,
            "breast_density": self.breast_density,
            "subtlety": self.subtlety,
            "bit_depth": self.bit_depth,
            "height": self.height,
            "width": self.width,
            "size": self.size,
            "aspect_ratio": self.aspect_ratio,
            "min_pixel_value": self.min_pixel_value,
            "max_pixel_value": self.max_pixel_value,
            "range_pixel_values": self.range_pixel_values,
            "mean_pixel_value": self.mean_pixel_value,
            "median_pixel_value": self.median_pixel_value,
            "std_pixel_value": self.std_pixel_value,
            "filepath": self.filepath,
            "fileset": self.fileset,
            "cancer": self.cancer,
        }
        return pd.DataFrame(data=d, index=[0])
