#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data_prep/meta.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday September 4th 2023 12:24:16 pm                                               #
# Modified   : Monday September 11th 2023 04:37:59 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import sys
import os
from datetime import datetime
from glob import glob
import logging

import pandas as pd
import pydicom

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------------------------------------ #
class MetaPrep:
    """Adds image metadata to the image file metadata

    Args:
        infilepath (str): The path to the mass or calc metadata file.
        outfilepath (str): The path to the output file.
    """

    __BASEDIR = "data/raw"

    def __init__(self, infilepath: str, outfilepath: str) -> None:
        self._infilepath = infilepath
        self._outfilepath = outfilepath
        self._input = pd.read_csv(infilepath)

    def prep_images(self) -> None:
        """Adds image metadata to the file metadata."""

        start = datetime.now()

        collection_meta = self._init_meta()

        fidx = 0

        if os.path.exists(self._outfilepath):
            x = input("Output file already exists. Delete? (y/n)")
            if "y" in x.lower():
                os.remove(self._outfilepath)
            else:
                exit()

        for idx, seriesmeta in self._input.iterrows():
            imagemeta = self._extract_metadata(seriesmeta)
            filepaths = self._get_filepaths(seriesmeta["file_location"])
            for filepath in filepaths:
                fidx += 1
                dicommeta = self._extract_dicom_data(filepath=filepath)
                collection_meta = self._combine_metadata(collection_meta, imagemeta, dicommeta)
            if (idx + 1) % 100 == 0:
                now = datetime.now()
                elapsed = (now - start).total_seconds()
                rrate = round((idx + 1) / elapsed, 2)
                irate = round(fidx / elapsed, 2)
                df = pd.DataFrame.from_dict(data=collection_meta, orient="columns")
                df.to_csv(path_or_buf=self._outfilepath, mode="a", index=False)
                collection_meta = self._init_meta()
                msg = f"Processed {idx+1} rows and {fidx} images in {round(elapsed,2)} seconds at {rrate} rows per second / {irate} images per second."
                logger.debug(msg)

        df = pd.DataFrame.from_dict(data=collection_meta, orient="columns")
        msg = f"Prepared dataframe of shape {df.shape}"
        logger.debug(msg)
        df.to_csv(path_or_buf=self._outfilepath, mode="a", index=False)

    def _init_meta(self) -> None:
        """Adds image metadata to the file metadata."""
        meta = {}
        meta["patient_id"] = []
        meta["subject_id"] = []
        meta["series_uid"] = []
        meta["description"] = []
        meta["view"] = []
        meta["side"] = []
        meta["casetype"] = []
        meta["fileset"] = []
        meta["filepath"] = []
        meta["width"] = []
        meta["height"] = []
        meta["aspect_ratio"] = []
        meta["bits"] = []
        meta["smallest_pixel"] = []
        meta["largest_pixel"] = []
        meta["pixel_range"] = []
        return meta

    def _extract_metadata(self, seriesmeta: pd.Series) -> dict:
        """Extracts metadata from the series"""
        meta = {}
        meta["patient_id"] = self._get_pid(seriesmeta["subject_id"])
        meta["subject_id"] = seriesmeta["subject_id"]
        meta["series_uid"] = seriesmeta["series_uid"]
        meta["description"] = seriesmeta["series_description"]
        meta["view"] = "MLO" if "MLO" in seriesmeta["subject_id"] else "CC"
        meta["side"] = "LEFT" if "LEFT" in seriesmeta["subject_id"] else "RIGHT"
        meta["casetype"] = seriesmeta["casetype"]
        meta["fileset"] = seriesmeta["fileset"]
        return meta

    def _get_filepaths(self, file_location: str) -> list[str]:
        """Returns filepaths within the file location
        Args:
            file_location (str): The file_location from a row within the metadata.csv file.
        """
        floc = file_location.replace("./CBIS", "CBIS")
        floc = os.path.join(self.__BASEDIR, floc)
        floc = floc + "**/*.dcm"
        return glob(floc, recursive=True)

    def _get_pid(self, subject_id: str) -> str:
        """Extracts the patient_id from the subject_id

        Args:
            subject_id (str): The subject_id from a row within the metadata.csv file.
        """
        sid = subject_id.split("_")
        pid = sid[1] + "_" + sid[2]
        return pid

    def _extract_dicom_data(self, filepath: str) -> dict:
        """Returns a dictionary of attributes from the dicom overlay file."""
        image = {}
        ds = pydicom.dcmread(filepath)
        image["filepath"] = filepath
        image["width"] = int(ds.Columns)
        image["height"] = int(ds.Rows)
        image["aspect_ratio"] = round(int(ds.Columns) / int(ds.Rows), 2)
        image["bits"] = int(ds.BitsStored)
        image["smallest_pixel"] = int(ds.SmallestImagePixelValue)
        image["largest_pixel"] = int(ds.LargestImagePixelValue)
        image["pixel_range"] = int(ds.LargestImagePixelValue) - int(ds.SmallestImagePixelValue)
        return image

    def _combine_metadata(self, collection_meta: dict, imagemeta: dict, dicommeta: dict) -> dict:
        """Combines the metadata into a dictionary of lists"""
        collection_meta["patient_id"].append(imagemeta["patient_id"])
        collection_meta["subject_id"].append(imagemeta["subject_id"])
        collection_meta["series_uid"].append(imagemeta["series_uid"])
        collection_meta["description"].append(imagemeta["description"])
        collection_meta["view"].append(imagemeta["view"])
        collection_meta["side"].append(imagemeta["side"])
        collection_meta["casetype"].append(imagemeta["casetype"])
        collection_meta["fileset"].append(imagemeta["fileset"])
        collection_meta["filepath"].append(dicommeta["filepath"])
        collection_meta["width"].append(dicommeta["width"])
        collection_meta["height"].append(dicommeta["height"])
        collection_meta["aspect_ratio"].append(dicommeta["aspect_ratio"])
        collection_meta["bits"].append(dicommeta["bits"])
        collection_meta["smallest_pixel"].append(dicommeta["smallest_pixel"])
        collection_meta["largest_pixel"].append(dicommeta["largest_pixel"])
        collection_meta["pixel_range"].append(dicommeta["pixel_range"])
        return collection_meta
