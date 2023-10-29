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
# Modified   : Sunday October 29th 2023 05:10:53 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import pytest
from dotenv import load_dotenv
from tqdm import tqdm

from bcd.config import Config
from bcd.container import BCDContainer
from bcd.core.orchestration.task import Task
from bcd.dal.io.image import ImageIO
from bcd.preprocess.image.convert import ImageConverter, ImageConverterParams

# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["data/**/*.*"]
# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #
IMAGE_FP = "data/meta/2_clean/dicom.csv"
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=redefined-outer-name
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
#                                  SET MODE TO TEST                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def mode():
    """Sets the mode to test"""
    prior_mode = Config.get_mode()
    Config.set_mode("test")
    yield
    Config.set_mode(prior_mode)


# ------------------------------------------------------------------------------------------------ #
#                             SET LOGGING LEVEL TO DEBUG                                           #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def log_level():
    """Sets the log level to DEBUG"""
    prior_level = Config.get_log_level()
    Config.set_log_level("DEBUG")
    yield
    Config.set_log_level(prior_level)


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
            "bcd.preprocess.image",
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
#                                          TASKS                                                   #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def tasks():
    """Creates a series of Task objects"""
    tasklist = []
    params = ImageConverterParams()
    for _ in range(5):
        task = Task.create(application=ImageConverter, params=params)
        tasklist.append(task)
    return tasklist


# ------------------------------------------------------------------------------------------------ #
#                                            IMAGES                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def images(container):
    image_list = []
    transformer = ["P1", "P2"]
    io = ImageIO()
    factory = container.dal.image_factory()
    df = pd.read_csv(IMAGE_FP)
    df = df.loc[df["series_description"] == "full mammogram images"]
    df = df.sample(n=10)
    i = 0
    for _, meta in tqdm(df.iterrows(), total=df.shape[0]):
        pixel_data = io.read(filepath=meta["filepath"])
        stage_id = i % 2
        image = factory.create(
            case_id=meta["case_id"],
            stage_id=stage_id,
            pixel_data=pixel_data,
            transformer=transformer[stage_id],
            task_id="standard_test_task_id" + str(stage_id),
        )
        i += 1
        image_list.append(image)
    return image_list
