#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/transfer.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday January 13th 2024 06:52:57 pm                                              #
# Modified   : Saturday January 13th 2024 07:43:45 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
import tensorflow as tf
from bcd import DataClass
# ------------------------------------------------------------------------------------------------ #
@dataclass
class FineTuneSchedule:
    max_epochs: int = 100
    fine_tune_epochs: int = 100
    early_stop_monitor
# ------------------------------------------------------------------------------------------------ #
class FineTuner:
    def __init__(self, max_epochs: int = 100, fine_tune_epochs: int = 10, early_stop: tf.keras.callbacks.EarlyStopping,