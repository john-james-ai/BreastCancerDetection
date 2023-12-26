#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/pectoral.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday December 24th 2023 11:04:18 pm                                               #
# Modified   : Monday December 25th 2023 09:13:24 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module for Pectoral Removal"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon

from bcd import Task


# ------------------------------------------------------------------------------------------------ #
class PectoralRemover(Task):
    def __init__(self, houghlines_max_threshold: int = 150):
        super().__init__()
        self._houghlines_max_threshold = houghlines_max_threshold
