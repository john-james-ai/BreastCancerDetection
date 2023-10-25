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
# Modified   : Wednesday October 25th 2023 06:12:17 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import pytest


from bcd.config import Config
from bcd.container import BCDContainer
from bcd.manage_data.io.image import ImageIO
from bcd.manage_data.entity.image import ImageFactory
from tqdm import tqdm

# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["data/**/*.*"]
# ------------------------------------------------------------------------------------------------ #
IMAGE_FP = "data/meta/2_clean/dicom.csv"


# ------------------------------------------------------------------------------------------------ #
#                                  SET MODE TO TEST                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def mode():
    config = Config()
    prior_mode = config.get_mode()
    config.set_mode(mode="test")
    yield
    config.set_mode(mode=prior_mode)


# ------------------------------------------------------------------------------------------------ #
#                             SET LOGGING LEVEL TO DEBUG                                           #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def log_level():
    config = Config()
    prior_level = config.get_log_level()
    config.set_log_level(level="DEBUG")
    yield
    config.set_log_level(level=prior_level)


# ------------------------------------------------------------------------------------------------ #
#                                DEPENDENCY INJECTION                                              #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def container():
    container = BCDContainer()
    container.init_resources()
    container.wire(
        packages=[
            "bcd.manage_data",
            "bcd.preprocess.convert",
            "bcd.preprocess.filter",
        ]
    )

    return container


# ------------------------------------------------------------------------------------------------ #
#                                         CASE IDS                                                 #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def case_ids():
    df = pd.read_csv(IMAGE_FP)
    df = df.loc[df["series_description"] == "full mammogram images"]
    df = df.sample(n=10)
    return list(df["case_id"])


# ------------------------------------------------------------------------------------------------ #
#                                            IMAGES                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def images():
    images = []
    preprocessors = ["TestPreprocessor1", "TestPreprocessor2"]
    io = ImageIO()
    factory = ImageFactory(case_fp=IMAGE_FP, config=Config)
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
            preprocessor=preprocessors[stage_id],
            task_id="standard_test_task_id" + str(stage_id),
        )
        i += 1
        images.append(image)
    return images
