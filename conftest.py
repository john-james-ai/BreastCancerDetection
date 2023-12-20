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
# Modified   : Wednesday December 20th 2023 04:53:42 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import cv2
import pandas as pd
import pytest

from bcd.config import Config
from bcd.container import BCDContainer

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
@pytest.fixture(scope="session", autouse=False)
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
        modules=["bcd.data.load"],
        packages=[
            "bcd.dal",
            "bcd.preprocess.image.flow",
            "bcd.preprocess.image.experiment",
        ],
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
#                                            EVALS                                                 #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def evals():
    return pd.read_csv(EVALUATION_FP)


# ------------------------------------------------------------------------------------------------ #
#                                           IMAGES                                                 #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def images():
    img1 = "data/image/1_dev/converted/train/benign/347c2455-cb62-40f8-a173-9e4eb9a21902.png"
    img2 = "data/image/1_dev/converted/train/benign/4ed91643-1e06-4b2c-8efb-bc60dd9e0313.png"
    img3 = "data/image/1_dev/converted/train/malignant/7dcc12fd-88f0-4048-a6ab-5dd0bd836f08.png"
    img4 = "data/image/1_dev/converted/train/malignant/596ef5db-9610-4f13-9c1a-4c411b1d957c.png"

    img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread(img4, cv2.IMREAD_GRAYSCALE)

    return [img1, img2, img3, img4]


# ------------------------------------------------------------------------------------------------ #
#                                        IMAGE META                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def image_meta():
    """Returns a dictionary containing a randomly selected case from the DICOM metadata dataset."""
    df = pd.read_csv(IMAGE_FP)
    s = df.sample(n=1)
    s = s.squeeze()
    return s.to_dict()
