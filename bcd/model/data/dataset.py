#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/data/dataset.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday January 9th 2024 12:44:48 am                                                #
# Modified   : Wednesday January 10th 2024 05:06:01 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member, line-too-long, import-error
# ------------------------------------------------------------------------------------------------ #
"""CBIS TensorFlow Dataset Module"""
import logging
import os
import shutil
from glob import glob
from typing import Callable

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from bcd.dal.image import ImageIO


# ------------------------------------------------------------------------------------------------ #
#                                CBIS TFRecords DATASET                                            #
# ------------------------------------------------------------------------------------------------ #
class CBISTFRecordsDataset:
    """Creates and provides access to the CBIS TensorFlow Dataset"""

    def __init__(
        self,
        metafilepath: str,
        destination: str,
        n: int = None,
        frac: float = None,
        condition: Callable = None,
        groupby: list = None,
        n_cpus: int = 12,
        shards_per_cpu: int = 10,
        random_state: int = None,
        force: (bool) = False,
    ) -> None:
        self._metafilepath = metafilepath
        self._destination = destination
        self._n = n
        self._frac = frac
        self._condition = condition
        self._groupby = groupby
        self._n_cpus = n_cpus
        self._shards_per_cpu = shards_per_cpu
        self._random_state = random_state
        self._force = force

        self._meta = None
        self._batchsize = None
        self._n_images = 0
        self._n_files = 0
        self._filenames = []

        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(logging.INFO)

    @property
    def destination(self) -> str:
        return self._destination

    @property
    def n_images(self) -> int:
        self._n_images = self._n_images or len(self._meta)
        return self._n_images

    @property
    def n_files(self) -> int:
        self._n_files = self._n_files or len(glob(self._destination + "*.tfrecords"))
        return self._n_files

    def create(self) -> None:
        """Creates a TFRecordDataset from DICOM image and metadata.

        Args:
            destination (str): Folder into which TFRecords will be stored.
            n (int): Number of images to sample.
            frac (float): Fraction of images to sample.
            condition (Callable): Lambda expression used to subset the data
            groupby (list): List of metadata elements for groupby sampling
            force (bool): Whether to overwrite data if TFRecords already exist at the destination.

        """
        self._validate()

        self._meta = self._select_images()

        self._batchsize = self._compute_batch_size()

        batches = [
            self._meta[i : i + self._batchsize]
            for i in range(0, len(self._meta), self._batchsize)
        ]
        self._n_files = len(batches)

        for i, batch in enumerate(tqdm(batches, total=len(batches))):
            # Create a TFRecord filepath for each batch.
            tfr_filepath = self._get_tfr_filepath(i)

            with tf.io.TFRecordWriter(tfr_filepath) as writer:
                # Write a TFRecord for each row in the batch.
                for _, row in batch.iterrows():
                    image_example = self._create_image_example(meta=row)
                    writer.write(image_example)

    def get_dataset(self) -> tf.data.Dataset:
        """Returns a raw TFRecordDataset object."""
        self._filenames = glob(self._destination + "*.tfrecords")
        self._n_files = len(self._filenames)

        if self._n_files == 0:
            msg = "TFRecords files have not been created."
            self._logger.warning(msg)
            raise UserWarning(msg)
        return tf.data.TFRecordDataset(
            self._filenames, num_parallel_reads=tf.data.AUTOTUNE
        )

    def parse_dataset(self) -> tf.data.Dataset:
        """Returns a parsed TensorFlow Dataset."""
        dataset = self.get_dataset()
        feature_description = {
            "fileset": tf.io.FixedLenFeature([], tf.string),
            "abnormality_type": tf.io.FixedLenFeature([], tf.string),
            "patient_id": tf.io.FixedLenFeature([], tf.string),
            "laterality": tf.io.FixedLenFeature([], tf.string),
            "image_view": tf.io.FixedLenFeature([], tf.string),
            "rows": tf.io.FixedLenFeature([], tf.int64),
            "cols": tf.io.FixedLenFeature([], tf.int64),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "image": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_image_function(proto):
            """Parse the input tf.train.Example proto using the dictionary above."""
            return tf.io.parse_single_example(proto, feature_description)

        return dataset.map(_parse_image_function)

    def _get_tfr_filepath(self, i: int) -> str:
        """Returns a filepath for the tfRecords file."""
        # Assumes no more than 999 shards. Actually, the number of shards is
        # 120 = 10*N where N=12 is the number of parallel processing nodes (cpus)
        shard_no = str(i).zfill(3)
        filename = "cbis_" + shard_no + ".tfrecords"
        return os.path.join(self._destination, filename)

    # -------------------------------------------------------------------------------------------- #
    def _create_image_example(self, meta: pd.Series) -> dict:
        """Creates a dictionary with features that may be relevant.

        Args:
            meta (pd.Series): Image metadata
        """
        image = ImageIO.read(meta["filepath"])

        feature = {
            "fileset": self._bytes_feature(tf.io.serialize_tensor(meta["fileset"])),
            "abnormality_type": self._bytes_feature(
                tf.io.serialize_tensor(meta["abnormality_type"][0:4])
            ),
            "patient_id": self._bytes_feature(
                tf.io.serialize_tensor(meta["patient_id"])
            ),
            "laterality": self._bytes_feature(
                tf.io.serialize_tensor(meta["laterality"])
            ),
            "image_view": self._bytes_feature(
                tf.io.serialize_tensor(meta["image_view"])
            ),
            "rows": self._int64_feature(meta["rows"]),
            "cols": self._int64_feature(meta["cols"]),
            "label": self._int64_feature(meta["cancer"]),
            "image": self._bytes_feature(tf.io.serialize_tensor(image)),
        }
        proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return proto.SerializeToString()

    # -------------------------------------------------------------------------------------------- #
    #    The following 4 methods converts a value to a type compatible with tf.train.Example.      #
    # -------------------------------------------------------------------------------------------- #
    def _bytes_feature(self, value: str):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = (
                value.numpy()
            )  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value: float):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _float_array_feature(self, value: float):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(self, value: int):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # -------------------------------------------------------------------------------------------- #
    def _validate(self) -> None:
        """Ensures the object is in a valid state.

        Protects against accidentally overwriting existing data
        and that required inputs exist.
        """
        if not os.path.exists(self._metafilepath):
            msg = f"Metadata filepath: {self._metafilepath} does not exist."
            self._logger.exception(msg)
            raise FileNotFoundError(msg)

        if os.path.exists(self._destination) and not self._force:
            msg = f"TFRecord Files already exist at {self._destination}. Aborting Dataset creation. To overwrite, set Force to True"
            self._logger.exception(msg)
            raise FileExistsError(msg)

        if self._n is not None and self._frac is not None:
            msg = "Ambiguous arguments. Both n and frac must not be non Null"
            self._logger.exception(msg)
            raise ValueError(msg)

        if not isinstance(self._n_cpus, (int, float)):
            msg = "n_cpus must be numeric."
            self._logger.exception(msg)
            raise TypeError(msg)

        if os.path.exists(self._destination) and self._force:
            shutil.rmtree(self._destination)

        os.makedirs(self._destination, exist_ok=True)

    # -------------------------------------------------------------------------------------------- #
    def _select_images(self) -> pd.DataFrame:
        """Returns a dataframe with metadata for the selected images."""
        df = pd.read_csv(self._metafilepath)

        if self._condition is not None:
            df = df[self._condition].copy(deep=True)

        if self._groupby is not None:
            df = df.groupby(by=self._groupby).sample(
                n=self._n, frac=self._frac, random_state=self._random_state
            )
        else:
            df = df.sample(n=self._n, frac=self._frac)

        df = df.drop_duplicates()

        self._n_images = df.shape[0]

        return df

    # -------------------------------------------------------------------------------------------- #
    def _compute_batch_size(self) -> int:
        """Sets batch size for TFRecord sharding

        Ideally, there should be 10*n_cpus TFRecord Files for optimized parallel I/O. At the same
        time, the individual TFRecord file should be at least ~10 MB+, ideally ~100 MB+. The
        batchsize specifies both the size and number of TFRecord files and is estimated
        as int(num_images / 10*n_cpus). The file size of each TFRecord is estimated as the
        sum of filesize / batch_size
        """

        n_files = self._shards_per_cpu * self._n_cpus
        batchsize = int(self._n_images / n_files)
        estimated_filesize = self._meta["file_size"].sum() / batchsize
        if estimated_filesize < (10 * 1024 * 1024):
            msg = f"Consider reducing the number of shards per cpu. Estimated filesize is less than 10 MB\n\tn_cpus: {self._n_cpus}\n\tshards_per_cpu: {self._shards_per_cpu}\n\tbatchsize: {batchsize}\n\tEstimated filesize {round(estimated_filesize,1)}"
            self._logger.exception(msg)
            raise ValueError(msg)
        return batchsize
