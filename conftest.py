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
# Modified   : Wednesday November 1st 2023 03:15:29 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from bcd.config import Config
from bcd.container import BCDContainer
from bcd.dal.io.image import ImageIO
from bcd.preprocess.image.evaluate import Evaluation
from bcd.preprocess.image.flow.state import Stage

# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["data/**/*.*", "bcd/preprocess/**/*.*"]
# ------------------------------------------------------------------------------------------------ #
IMAGE_FP = "data/meta/2_clean/dicom.csv"
# ------------------------------------------------------------------------------------------------ #
# pylint: disable=redefined-outer-name
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
    ctr.wire(modules=["bcd.preprocess.image.flow.convert"])

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
def images(container):
    image_list = []
    method = ["P1", "P2"]
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
            method=method[stage_id],
            task_id="standard_test_task_id" + str(stage_id),
        )
        i += 1
        image_list.append(image)
    return image_list


# ------------------------------------------------------------------------------------------------ #
#                                            EVALS                                                 #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def evals():
    rng = np.random.default_rng()
    eval_list = []
    stage_ids = [1, 2]
    method_names = ["MeanFilter", "MedianFilter", "GaussianFilter", "LintFilter", "Hairnet"]
    steps = ["Binarize", "LargestContour"]
    diag = [True, False]
    for i in range(50):
        e = Evaluation(
            image_uid=str(uuid4()),
            mode="test",
            stage_id=stage_ids[i % 2],
            stage=Stage(uid=stage_ids[i % 2]).name,
            step=steps[i % 2],
            method=method_names[i % 5],
            params="some_parameters",
            mse=rng.random() * 50,
            psnr=rng.random() * 20,
            ssim=rng.random(),
            image_view="CC",
            abnormality_type="mass",
            assessment=np.random.randint(6, size=1)[0],
            cancer=diag[i % 2],
            evaluated=datetime.now(),
        )
        eval_list.append(e)
    return eval_list
