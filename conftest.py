#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /conftest.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday September 22nd 2023 06:54:46 am                                              #
# Modified   : Friday October 27th 2023 01:11:55 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import pytest

from tqdm import tqdm

from bcd.config import Config
from bcd.container import BCDContainer
from bcd.infrastructure.io.image import ImageIO
from bcd.core.image.factory import ImageFactory
from bcd.preprocess.convert import ImageConverter, ImageConverterParams, ImageConverterTask
from bcd.preprocess.filter import FilterParams, FilterTask, MeanFilter, MedianFilter, GaussianFilter


# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["data/**/*.*"]
# ------------------------------------------------------------------------------------------------ #
IMAGE_FP = "data/meta/2_clean/dicom.csv"


# ------------------------------------------------------------------------------------------------ #
#                                  SET MODE TO TEST                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def mode():
    """Sets the mode to test"""
    config = Config()
    prior_mode = config.mode
    config.mode = "test"
    yield
    config.mode = prior_mode


# ------------------------------------------------------------------------------------------------ #
#                             SET LOGGING LEVEL TO DEBUG                                           #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def log_level():
    """Sets the log level to DEBUG"""
    config = Config()
    prior_level = config.log_level
    config.log_level = "DEBUG"
    yield
    config.log_level = prior_level


# ------------------------------------------------------------------------------------------------ #
#                                DEPENDENCY INJECTION                                              #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def container():
    """Wires the container."""
    ctr = BCDContainer()
    ctr.init_resources()
    ctr.wire(
        packages=[
            "bcd.core",
            "bcd.preprocess.convert",
            "bcd.preprocess.filter",
        ]
    )

    return ctr


# ------------------------------------------------------------------------------------------------ #
#                                         CASE IDS                                                 #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def case_ids():
    """Creates a list of case ids."""
    df = pd.read_csv(IMAGE_FP)
    df = df.loc[df["series_description"] == "full mammogram images"]
    df = df.sample(n=10)
    return list(df["case_id"])


# ------------------------------------------------------------------------------------------------ #
#                                            IMAGES                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def images():
    """Creates a list of images."""
    imgs = []
    preprocessors = ["P1", "P2", "P3"]
    io = ImageIO()
    factory = ImageFactory(case_fp=IMAGE_FP, config=Config)
    df = pd.read_csv(IMAGE_FP, low_memory=False, float_precision='high', dtype={'cancer': bool})
    df = df.loc[df["series_description"] == "full mammogram images"]
    df = df.sample(n=15)
    i = 0
    for _, meta in tqdm(df.iterrows(), total=df.shape[0]):
        pixel_data = io.read(filepath=meta["filepath"])
        stage_uid = i % 3
        image = factory.create(
            case_id=meta["case_id"],
            stage_uid=stage_uid,
            pixel_data=pixel_data,
            preprocessor=preprocessors[i%3],
            task_id="standard_test_task_id" + str(stage_uid),
        )
        i += 1
        imgs.append(image)
    return imgs


# ------------------------------------------------------------------------------------------------ #
#                                            TASKS                                                 #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def tasks():
    """Creates a list of tasks."""
    tasklist = []
    tasks_ = [ImageConverterTask, FilterTask, FilterTask, FilterTask]
    apps = [ImageConverter, MeanFilter, MedianFilter, GaussianFilter]
    params_ = [ImageConverterParams(), FilterParams(), FilterParams(), FilterParams()]

    for i in range(5):
        params = params_[i % 4]
        app = apps[i % 4]
        task_ = tasks_[i % 4]
        task = task_.create(application=app, params=params, config=Config)
        tasklist.append(task)
    return tasklist
