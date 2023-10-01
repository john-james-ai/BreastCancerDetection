#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/utils/string.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 1st 2023 01:02:34 pm                                                 #
# Modified   : Sunday October 1st 2023 01:17:55 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import string

ABBREVIATIONS = {"calc ": "calcification "}


def proper(s: str) -> str:
    s = s.replace("_", " ")
    for abbr, full in ABBREVIATIONS.items():
        s = s.replace(abbr, full)
    s = string.capwords(s)
    return s
