#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_manage_data/test_repo/test_image_repo.py                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 22nd 2023 02:26:44 am                                                #
# Modified   : Wednesday October 25th 2023 11:53:18 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import inspect
from datetime import datetime
import pytest
import logging

import pandas as pd

from bcd.core.image.entity import Image

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.repo
@pytest.mark.image_repo
class TestImageRepo:  # pragma: no cover
    # ============================================================================================ #
    def test_setup(self, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        try:
            repo.delete_by_mode()
        except Exception:
            pass
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_add_exists(self, images, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        for image in images:
            # Add image to repository
            repo.add(image=image)
            # Confirm image has been saved to disk
            assert os.path.exists(image.filepath)
            # Confirm image exists in repo.
            assert repo.exists(id=image.id)
            # Test adding image already exists
            with pytest.raises(FileExistsError):
                repo.add(image=image)

        assert repo.count() == 10
        assert not repo.exists(id="999")

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get(self, images, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        for image in images:
            image2 = repo.get(id=image.id)
            assert image == image2

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_by_stage(self, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        image_meta, images = repo.get_by_stage(stage_id=0)
        assert len(image_meta) == 5
        assert len(images) == 5
        for id, image in images.items():
            assert image.stage_id == 0
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Sampling n
        image_meta, images = repo.get_by_stage(stage_id=1, n=3)
        assert len(image_meta) == 3
        assert len(images) == 3
        for id, image in images.items():
            assert image.stage_id == 1
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Sampling frac
        image_meta, images = repo.get_by_stage(stage_id=1, frac=0.5)
        assert len(image_meta) == 2
        assert len(images) == 2
        for id, image in images.items():
            assert image.stage_id == 1
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Exception
        with pytest.raises(FileNotFoundError):
            image_meta, images = repo.get_by_stage(stage_id=99, frac=0.5)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_by_preprocessor(self, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        image_meta, images = repo.get_by_preprocessor(preprocessor="TestPreprocessor1")
        assert len(image_meta) == 5
        assert len(images) == 5
        for id, image in images.items():
            assert image.stage_id == 0
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None
            assert image.preprocessor == "TestPreprocessor1"

        # Test Sampling n
        image_meta, images = repo.get_by_preprocessor(preprocessor="TestPreprocessor1", n=3)
        assert len(image_meta) == 3
        assert len(images) == 3
        for id, image in images.items():
            assert image.stage_id == 0
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None
            assert image.preprocessor == "TestPreprocessor1"

        # Test Sampling frac
        image_meta, images = repo.get_by_preprocessor(preprocessor="TestPreprocessor2", frac=0.5)
        assert len(image_meta) == 2
        assert len(images) == 2
        for id, image in images.items():
            assert image.stage_id == 1
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None
            assert image.preprocessor == "TestPreprocessor2"

        # Test Exception
        with pytest.raises(FileNotFoundError):
            image_meta, images = repo.get_by_preprocessor(preprocessor="Invalid")
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_by_mode(self, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        image_meta, images = repo.get_by_mode()
        assert len(image_meta) == 10
        assert len(images) == 10
        for id, image in images.items():
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Sampling n
        image_meta, images = repo.get_by_mode(n=3)
        assert len(image_meta) == 6
        assert len(images) == 6
        for id, image in images.items():
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # Test Sampling frac
        image_meta, images = repo.get_by_mode(frac=0.5)
        assert len(image_meta) == 4
        assert len(images) == 4
        for id, image in images.items():
            assert isinstance(id, str)
            assert isinstance(image, Image)
            assert image.pixel_data is not None

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get_meta(self, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
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
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_count(self, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        assert repo.count() == 10
        condition = lambda df: df["stage_id"] == 1
        assert repo.count(condition) == 5
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_delete_stage(self, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        repo.delete_by_stage(stage_id=0)
        assert repo.count() == 5

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_delete_by_preprocessor(self, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        repo.delete_by_preprocessor(preprocessor="TestPreprocessor2")
        assert repo.count() == 0
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_delete_by_mode(self, images, container, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        repo = container.repo.image()
        for image in images:
            # Add image to repository
            repo.add(image=image)
        repo.delete_by_mode()
        assert repo.count() == 0
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
