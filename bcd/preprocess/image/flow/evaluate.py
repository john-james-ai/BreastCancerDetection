#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/evaluate.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 27th 2023 03:24:36 am                                                #
# Modified   : Monday November 6th 2023 06:30:38 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from tqdm import tqdm

from bcd.dal.repo.uow import UoW
from bcd.preprocess.image.method.evaluate import Evaluation


# ------------------------------------------------------------------------------------------------ #
#                                   EVALUATOR                                                      #
# ------------------------------------------------------------------------------------------------ #
class Evaluator:
    """Conducts Evaluations of Experiments"""

    def __init__(self, stage_id: int, uow: UoW) -> None:
        self._stage_id = stage_id
        self._uow = uow

    def run(self) -> None:
        # Obtain the image and eval repositories
        image_repo = self._uow.image_repo
        eval_repo = self._uow.eval_repo

        # Extract the stage 0 image metadata
        condition_orig = lambda df: df["stage_id"] == 0
        orig_meta = image_repo.get_meta(condition=condition_orig)

        # Extract metadata for the test stage of interest
        condition_test = lambda df: df["stage_id"] == self._stage_id
        test_meta = image_repo.get_meta(condition=condition_test)
        # Group the tests by case_id which links the test case to the original
        # stage 0 image.
        cases = test_meta.groupby(by="case_id")
        for case_id, group in tqdm(cases, desc="case", total=len(cases)):
            # Obtain the original image for the case
            orig_uid = orig_meta.loc[orig_meta["case_id"] == case_id]["uid"].values[0]
            orig_image = image_repo.get(uid=orig_uid)
            # Iterate through the tests of various methods performed
            # on the origin image.
            for _, image_meta in tqdm(group.iterrows(), desc="tests", total=(len(group))):
                # Obtain the test image
                test_image = image_repo.get(uid=image_meta["uid"])
                # Evaluate the test image vis-a-vis the original.
                ev = Evaluation.evaluate(orig=orig_image, test=test_image, method=test_image.method)
                # Persist the evaluation.
                eval_repo.add(evaluation=ev)
