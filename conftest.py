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
# Modified   : Sunday October 29th 2023 01:51:50 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import pytest
from dotenv import load_dotenv

from bcd.config import Config
from bcd.container import BCDContainer

# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["data/**/*.*", "bcd/core/**/*.*"]
# ------------------------------------------------------------------------------------------------ #
load_dotenv()
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
