#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/artifact.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday February 8th 2024 11:08:26 pm                                              #
# Modified   : Thursday February 8th 2024 11:23:04 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Artifact Module"""
from dataclasses import dataclass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ModelArtifact:
    """Value object encapsulating identification of a model artifact"""

    name: str
    version: str
    dataset: str

    @property
    def id(self) -> str:
        return self.name + "_" + self.version + "-" + self.dataset
