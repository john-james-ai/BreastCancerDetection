#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 05:01:26 pm                                              #
# Modified   : Sunday October 22nd 2023 12:02:54 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from datetime import datetime
from uuid import uuid4

from bcd.manage_data.io.file import IOService
from bcd.manage_data import STAGES


# ------------------------------------------------------------------------------------------------ #
class ImagePreprocessor:
    def __init__(self, config_filepath: str) -> None:
        config_filepath = os.path.abspath(config_filepath)
        self._config = IOService.read(config_filepath)

    def execute(self) -> None:
        self._convert_png()

    def _convert_png(self) -> None:
        """Converts DICOM images to PNG format"""
        # Load the metadata
        dicom_meta = IOService.read(self._config["convert_png"]["dicom_meta_filepath"])
        # Extract full mammogram images
        dicom_meta = dicom_meta.loc[dicom_meta["series_description"] == "full mammogram images"]
        # Extract columns of interest
        columns = [
            "id",
            "filepath",
            "height",
            "width",
            "size",
            "aspect_ratio",
            "bit_depth",
            "min_pixel_value",
            "max_pixel_value",
            "range_pixel_values",
            "mean_pixel_value",
            "median_pixel_value",
            "std_pixel_value",
            "case_id",
            "fileset",
            "cancer",
        ]
        image_meta = dicom_meta[columns]
        image_meta["mode"] = "dev"
        image_meta["stage_id"] = 0
        image_meta["stage"] = STAGES[0]
        image_meta["created"] = datetime.now()
        image_meta["task_run"] = str(uuid4())

        # Update the ids for the images
        image_meta["id"].apply(lambda x: str(uuid4()))
        # Update filepath
        directory = self._config["convert_png"]["dest_dir"]
        image_meta["filepath"] = image_meta["id"] + ".png"
        image_meta["filepath"].apply(lambda x: os.path.join(directory, image_meta["filepath"]))
