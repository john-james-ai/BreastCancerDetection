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
# Modified   : Monday November 6th 2023 06:14:40 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import cv2
import pandas as pd
import pytest
from tqdm import tqdm

from bcd.config import Config
from bcd.container import BCDContainer
from bcd.image import ImageFactory

# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["data/**/*.*", "bcd/preprocess/**/*.*"]
# ------------------------------------------------------------------------------------------------ #
IMAGE_FP = "data/meta/2_clean/dicom.csv"
EVALUATION_FP = "tests/data/3_denoise/results.csv"
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=redefined-outer-name, no-member
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
#                                    CURRENT MODE                                                  #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def current_mode():
    return Config.get_mode()


# ------------------------------------------------------------------------------------------------ #
#                                  SET MODE TO TEST                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def mode():
    prior_mode = Config.get_mode()
    Config.set_mode(mode="test")
    yield
    Config.set_mode(mode=prior_mode)


# ------------------------------------------------------------------------------------------------ #
#                             SET LOGGING LEVEL TO DEBUG                                           #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=True)
def log_level():
    """Sets the log level to DEBUG"""
    prior_level = Config.get_log_level()
    Config.set_log_level("DEBUG")
    yield
    Config.set_log_level(prior_level)


# ------------------------------------------------------------------------------------------------ #
#                                DEPENDENCY INJECTION                                              #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def container():
    """Wires the container."""
    ctr = BCDContainer()
    ctr.init_resources()
    ctr.wire(
        modules=["bcd.etl.load"],
        packages=["bcd.dal", "bcd.preprocess.image.flow", "bcd.preprocess.image.experiment"],
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
    image_list = []
    method = ["P1", "P2"]
    factory = ImageFactory()
    df = pd.read_csv(IMAGE_FP)
    df = df.loc[df["series_description"] == "full mammogram images"]
    df = df.sample(n=10)
    i = 0
    for _, meta in tqdm(df.iterrows(), total=df.shape[0]):
        pixel_data = cv2.imread(meta["filepath"], cv2.IMREAD_UNCHANGED)
        stage_id = i % 2
        image = factory.create(
            case_id=meta["case_id"],
            stage_id=stage_id,
            pixel_data=pixel_data,
            method=method[stage_id],
            build_time=22,
            task_id="some_task_id",
        )
        i += 1
        image_list.append(image)
    return image_list


# ------------------------------------------------------------------------------------------------ #
#                                            EVALS                                                 #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def evals():
    return pd.read_csv(EVALUATION_FP)
