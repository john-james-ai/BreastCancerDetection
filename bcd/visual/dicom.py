#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/visual/dicom.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday September 4th 2023 04:52:12 am                                               #
# Modified   : Monday September 4th 2023 11:48:10 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module for visualizing DICOM Mammography"""
from __future__ import annotations
import os

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pydicom


# ------------------------------------------------------------------------------------------------ #
class DICOMVisualizer:
    """Visualizes DICOM Images

    Args:
        case_metadata_filepath (str): Metadata for calcification and mass training and test cases.
        max_width (int): The maximum number of images to render in a single row.

    """

    def __init__(
        self,
        case_metadata_filepath: str,
        max_width: int = 5,
        max_images: int = 20,
        row_height: int = 4,
    ) -> None:
        self._case_metadata_filepath = case_metadata_filepath
        self._cases = pd.read_csv(self._case_metadata_filepath)
        self._max_width = max_width
        self._max_images = max_images
        self._row_height = row_height
        self._count = max_images
        self._subset = self._cases

    def reset(self) -> None:
        self._subset = self._cases

    def add_abnormality_type_filter(self, abnormality_type: str) -> DICOMVisualizer:
        """Filters the data by abnormality type

        Args:
        """
        abnormality_type = abnormality_type.lower()
        self._subset = self._subset.loc[self._subset["abnormality_type"] == abnormality_type]
        return self

    def add_density_filter(self, density: int) -> DICOMVisualizer:
        """Filters the dataset by density

        Args:
            density (int): A BI-RADS density category in [1,4]
        """
        self._subset = self._subset.loc[self._subset["breast density"] == density]
        return self

    def add_side_filter(self, side: str) -> DICOMVisualizer:
        """Filters the data by left or right side breast

        Args:
            side (str): Either 'LEFT' or 'RIGHT'
        """
        side = side.upper()
        self._subset = self._subset.loc[self._subset["left or right breast"] == side]
        return self

    def add_view_filter(self, view: str) -> DICOMVisualizer:
        """Filters the dataset by view

        Args:
            view (str): Either 'CC' or 'MLO'
        """
        view = view.upper()
        self._subset = self._subset.loc[self._subset["image view"] == view]
        return self

    def add_assessment_filter(self, assessment: int) -> DICOMVisualizer:
        """Filters the dataset by BI-RADS assessment

        Args:
            assessment (int): A BI-RADS assessment score in [0,5]
        """
        assessment = int(assessment)
        self._subset = self._subset.loc[self._subset["assessment"] == assessment]
        return self

    def add_subtlety_filter(self, subtlety: int) -> DICOMVisualizer:
        """Filters the dataset by subtlety

        Args:
            subtlety (int): Value in range [1,5]
        """
        subtlety = int(subtlety)
        self._subset = self._subset.loc[self._subset["subtlety"] == subtlety]
        return self

    def add_pathology_filter(self, pathology: int) -> DICOMVisualizer:
        """Filters the dataset by pathology

        Args:
            pathology (int): 0 means benign, or 1 meaning malignant.
        """
        pathology = int(pathology)
        self._subset = self._subset.loc[self._subset["cancer"] == pathology]
        return self

    def add_count_filter(self, count: int) -> DICOMVisualizer:
        self._count = count
        return self

    def show(self) -> None:
        images = self._select_images()
        rows, cols = self._get_canvas_size(images)
        fig = plt.figure(figsize=(12, rows * self._row_height))

        idx = 0
        for image in tqdm(images):
            idx += 1
            fig.add_subplot(rows, cols, idx)
            filepath = os.path.join(image["location"], image["filename"])
            img = pydicom.dcmread(filepath)
            plt.imshow(img.pixel_array, cmap="gray_r")
            title = (
                "Patient Id: "
                + image["patient_id"]
                + " Abnormality Type: "
                + image["abnormality type"]
                + " Breast Density: "
                + image["density"]
                + "Side: "
                + image["side"]
                + "\nView: "
                + image["image view"]
                + " Subtlety: "
                + image["subtlety"]
                + " Assessment "
                + image["assessment"]
                + " Pathology: "
                + image["pathology"]
            )
            plt.title(title)
            plt.show()

    def _select_images(self) -> list:
        image_meta = self._subset.sample(n=self._count, replace=False)
        image_data = []
        for _, image in image_meta.iterrows():
            image_data.append(image)
        return image_data

    def _get_canvas_size(self, images: list) -> tuple:
        n_images = len(images)
        for i in range(self._max_width):
            width = self._max_width - i
            if n_images % width == 0:
                rows = n_images / width
                cols = width
                return rows, cols
