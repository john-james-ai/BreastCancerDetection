#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_dal/test_repo/test_image_repo.py                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 02:26:44 am                                                #
# Modified   : Sunday November 5th 2023 12:54:06 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
import os
from datetime import datetime

import pandas as pd
import pytest

from bcd.config import Config
from bcd.image import Image

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, redefined-builtin, broad-exception-caught
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.repo
@pytest.mark.image_repo
class TestImageRepo:  # pragma: no cover
    # ============================================================================================ #
    def test_mode(self, current_mode):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        if current_mode != "test":
            msg = "\nCHANGE MODE TO TEST BEFORE RUNNING PYTEST!\nExiting pytest!\n"
            logger.exception(msg)
            pytest.exit(msg)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_setup(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        repo.delete_by_mode()
        registry_filepath = os.path.join(Config.get_data_dir(), "registry.csv")
        try:
            os.remove(registry_filepath)
        except Exception:
            pass

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_add_exists(self, images, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        for image in images:
            # Add image to repository
            repo.add(image=image)
            # Confirm image has been saved to disk
            assert os.path.exists(image.filepath)
            # Confirm image exists in repo.
            assert repo.exists(uid=image.uid)
            # Test adding image already exists
            with pytest.raises(FileExistsError):
                repo.add(image=image)

        assert repo.count() == 10
        assert not repo.exists(uid="999")

        registry_filepath = os.path.join(Config.get_data_dir(), "registry.csv")
        assert os.path.exists(registry_filepath)
        registry = pd.read_csv(registry_filepath)
        assert len(registry) == 10

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get(self, images, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        for image in images:
            image2 = repo.get(uid=image.uid)
            assert isinstance(image2, Image)
            logger.debug(f"\nImage Difference: \n{image.difference(image2)}")
            # assert image == image2

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_by_stage(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        image_meta, images = repo.get_by_stage(stage_id=0)
        assert len(image_meta) == 5
        assert len(images) == 5
        for id, image in images.items():
            assert image.stage_id == 0
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Exception
        with pytest.raises(FileNotFoundError):
            image_meta, images = repo.get_by_stage(stage_id=99)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_by_method(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        image_meta, images = repo.get_by_method(method="P1")
        assert len(image_meta) == 5
        assert len(images) == 5
        for id, image in images.items():
            # assert image.stage_id == 0
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None
            assert image.method == "P1"

        # Test Exception
        with pytest.raises(FileNotFoundError):
            image_meta, images = repo.get_by_method(method="Invalid")
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_by_mode(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        image_meta, images = repo.get_by_mode()
        assert len(image_meta) == 10
        assert len(images) == 10
        for id, image in images.items():
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_meta(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        meta = repo.get_meta()
        assert len(meta) == 10
        assert isinstance(meta, pd.DataFrame)
        logger.debug(f"Test Get Image Meta:\n{meta}")

        condition = lambda df: (df["stage_id"] == 0) & (df["mode"] == "test")
        meta = repo.get_meta(condition=condition)
        assert len(meta) == 5

        condition = lambda df: (df["stage_id"] == 99) & (df["mode"] == "test")
        meta = repo.get_meta(condition=condition)
        assert len(meta) == 0

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_count(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        assert repo.count() == 10
        condition = lambda df: df["stage_id"] == 1
        assert repo.count(condition) == 5
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_sample(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        metadata, images = repo.sample(n=5)
        assert len(metadata) == 5
        assert isinstance(metadata, pd.DataFrame)
        assert isinstance(images, dict)
        for _, image in images.items():
            assert isinstance(image, Image)

        metadata, images = repo.sample(frac=0.5)
        assert len(metadata) == 5
        assert isinstance(metadata, pd.DataFrame)
        assert isinstance(images, dict)
        for _, image in images.items():
            assert isinstance(image, Image)

        metadata, images = repo.sample(n=2, groupby="method")
        assert len(metadata) == 4
        assert isinstance(metadata, pd.DataFrame)
        assert isinstance(images, dict)
        for _, image in images.items():
            assert isinstance(image, Image)

        metadata, images = repo.sample(frac=0.5, groupby="method")
        assert len(metadata) == 4
        assert isinstance(metadata, pd.DataFrame)
        assert isinstance(images, dict)
        for _, image in images.items():
            assert isinstance(image, Image)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_delete_stage(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        repo.delete_by_stage(stage_id=0)
        assert repo.count() == 5

        registry_filepath = os.path.join(Config.get_data_dir(), "registry.csv")
        assert os.path.exists(registry_filepath)
        registry = pd.read_csv(registry_filepath)
        assert len(registry) == 5

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_delete_by_method(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        repo.delete_by_method(method="P2")
        assert repo.count() == 0
        registry_filepath = os.path.join(Config.get_data_dir(), "registry.csv")
        assert os.path.exists(registry_filepath)
        registry = pd.read_csv(registry_filepath)
        assert len(registry) == 0
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_delete_by_mode(self, images, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        for image in images:
            # Add image to repository
            repo.add(image=image)

        assert repo.count() == 10
        registry_filepath = os.path.join(Config.get_data_dir(), "registry.csv")
        assert os.path.exists(registry_filepath)
        registry = pd.read_csv(registry_filepath)
        assert len(registry) == 10
        assert not repo.exists(uid="999")
        repo = container.dal.image_repo()
        repo.delete_by_mode()
        assert repo.count() == 0
        registry = pd.read_csv(registry_filepath)
        assert len(registry) == 0
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
