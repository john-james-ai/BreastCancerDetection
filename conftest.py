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
# Modified   : Monday October 23rd 2023 12:10:06 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import dotenv
import datetime
from uuid import uuid4
import pandas as pd
import pytest

from bcd.container import BCDContainer
from bcd.preprocess.base import TaskRun
from bcd.manage_data import STAGES

# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["data/**/*.*"]
# ------------------------------------------------------------------------------------------------ #
IMAGE_FP = "data/meta/2_clean/dicom.csv"


# ------------------------------------------------------------------------------------------------ #
#                                  SET MODE TO TEST                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def mode():
    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)
    prior_mode = os.environ["MODE"]
    os.environ["MODE"] = "test"
    dotenv.set_key(dotenv_file, "MODE", os.environ["MODE"])
    yield
    os.environ["MODE"] = prior_mode
    dotenv.set_key(dotenv_file, "MODE", os.environ["MODE"])


# ------------------------------------------------------------------------------------------------ #
#                                DEPENDENCY INJECTION                                              #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def container():
    container = BCDContainer()
    container.init_resources()
    container.wire(packages=["bcd.manage_data"])

    return container


# ------------------------------------------------------------------------------------------------ #
#                                         IMAGE                                                    #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def case_ids():
    df = pd.read_csv(IMAGE_FP)
    df = df.loc[df["series_description"] == "full mammogram images"]
    df = df.sample(n=10)
    return list(df["case_id"])


# ------------------------------------------------------------------------------------------------ #
#                                        TASKRUN                                                   #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def taskruns():
    taskruns = []
    params = [
        {"n": 10},
        {"kernel": 20, "a": 20},
        {"b": 20, "c": 20, "d": 200},
        {"e": 20},
        {"b": 20, "f": 20, "d": 200},
        {"b": 20, "c": 20, "kernel": 200},
        {"x": 20, "y": 20, "d": 200, "z": "a"},
        {"w": 20, "f": 20, "d": 200},
        {"x": 20, "c": 20, "kernel": 200},
        {"y": 20, "hat": 20, "d": 200, "z": "a"},
    ]
    for i in range(10):
        taskid = i % 2
        started = datetime.datetime.now() - datetime.timedelta(seconds=20)
        ended = datetime.datetime.now()
        duration = (ended - started).total_seconds()
        images_processed = 100
        image_processing_time = duration / images_processed
        tr = TaskRun(
            id=str(uuid4()),
            task="TestTaskRunRepo" + str(taskid),
            mode="test",
            stage_id=0,
            stage=STAGES[0],
            started=started,
            ended=ended,
            duration=duration,
            images_processed=images_processed,
            image_processing_time=image_processing_time,
            success=True,
            params=params[i],
        )
        taskruns.append(tr)
    return taskruns
