#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/__init__.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 31st 2023 07:36:52 pm                                               #
# Modified   : Thursday September 28th 2023 02:25:00 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
CORE_VARIABLES = [
    "patient_id",
    "breast_density",
    "left_or_right_breast",
    "image_view",
    "abnormality_id",
    "abnormality_type",
    "assessment",
    "pathology",
    "subtlety",
    "fileset",
    "case_id",
    "cancer",
]
CALC_VARIABLES = [
    "patient_id",
    "breast_density",
    "left_or_right_breast",
    "image_view",
    "abnormality_id",
    "abnormality_type",
    "calc_type",
    "calc_distribution",
    "assessment",
    "pathology",
    "subtlety",
    "fileset",
    "case_id",
    "cancer",
]

MASS_VARIABLES = [
    "patient_id",
    "breast_density",
    "left_or_right_breast",
    "image_view",
    "abnormality_id",
    "abnormality_type",
    "assessment",
    "pathology",
    "subtlety",
    "fileset",
    "mass_shape",
    "mass_margins",
    "case_id",
    "cancer",
]
