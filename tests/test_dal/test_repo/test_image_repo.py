#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_dal/test_image_repo.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 02:26:44 am                                                #
# Modified   : Saturday October 28th 2023 08:56:21 pm                                              #
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

from bcd.core.image.entity import Image

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, redefined-builtin, broad-exception-caught
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.repo
@pytest.mark.image_repo
class TestImageRepo:  # pragma: no cover
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
        try:
            repo.delete_by_mode()
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

        assert repo.count() == 15
        assert not repo.exists(uid="999")

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
            logger.debug(image)
            logger.debug(image2)
            assert image == image2

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
        image_meta, images = repo.get_by_stage(stage_uid=0)
        assert len(image_meta) < 8
        assert len(images) < 8
        for id, image in images.items():
            assert image.stage_uid == 0
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Sampling n
        image_meta, images = repo.get_by_stage(stage_uid=1, n=2)
        assert len(image_meta) == 2
        assert len(images) == 2
        for id, image in images.items():
            assert image.stage_uid == 1
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Sampling frac
        image_meta, images = repo.get_by_stage(stage_uid=1, frac=0.5)
        assert len(image_meta) < 5
        assert len(images) < 5
        for id, image in images.items():
            assert image.stage_uid == 1
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Exception
        with pytest.raises(FileNotFoundError):
            image_meta, images = repo.get_by_stage(stage_uid=99, frac=0.5)

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
    def test_get_by_transformer(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        image_meta, images = repo.get_by_transformer(preprocessor="P1")
        assert len(image_meta) == 5
        assert len(images) == 5
        for id, image in images.items():
            # assert image.stage_uid == 0
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None
            assert image.preprocessor == "P1"

        # Test Sampling n
        image_meta, images = repo.get_by_transformer(preprocessor="P1", n=3)
        assert len(image_meta) == 3
        assert len(images) == 3
        for id, image in images.items():
            # assert image.stage_uid == 0
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None
            assert image.preprocessor == "P1"

        # Test Sampling frac
        image_meta, images = repo.get_by_transformer(preprocessor="P2", frac=0.5)
        assert len(image_meta) == 2
        assert len(images) == 2
        for id, image in images.items():
            # assert image.stage_uid == 1
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None
            assert image.preprocessor == "P2"

        # Test Exception
        with pytest.raises(FileNotFoundError):
            image_meta, images = repo.get_by_transformer(preprocessor="Invalid")
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
        assert len(image_meta) == 15
        assert len(images) == 15
        for id, image in images.items():
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Sampling n
        image_meta, images = repo.get_by_mode(n=2)
        assert len(image_meta) > 2
        assert len(images) > 2
        for id, image in images.items():
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Sampling frac
        image_meta, images = repo.get_by_mode(frac=0.5)
        assert len(image_meta) > 2
        assert len(images) > 2
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
        assert len(meta) == 15
        assert isinstance(meta, pd.DataFrame)
        logger.debug(f"Test Get Image Meta:\n{meta}")

        condition = lambda df: (df["stage_uid"] == 0) & (df["mode"] == "test")
        meta = repo.get_meta(condition=condition)
        assert len(meta) == 5

        condition = lambda df: (df["stage_uid"] == 99) & (df["mode"] == "test")
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
        assert repo.count() == 15
        condition = lambda df: df["stage_uid"] == 1
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
    def test_delete_stage(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        repo.delete_by_stage(stage_uid=0)
        assert repo.count() == 10

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
    def test_delete_by_transformer(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        repo.delete_by_transformer(preprocessor="P2")
        assert repo.count() == 5
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
    def test_delete_by_mode(self, container):
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at \
                {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.dal.image_repo()
        repo.delete_by_mode()
        assert repo.count() == 0
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in \
                {duration} seconds at {start.strftime('%I:%M:%S %p')} on \
                    {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)