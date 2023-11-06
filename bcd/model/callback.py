#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/callback.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday November 5th 2023 11:35:52 am                                                #
# Modified   : Sunday November 5th 2023 03:32:20 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""TensorFlow Callback Module """
from datetime import datetime

from keras.callbacks import Callback

# ------------------------------------------------------------------------------------------------ #


class DurationCallback(Callback):
    """Records duration of each epoch in the model history."""

    def __init__(self, name: str = "duration"):
        super().__init__()
        self._start = None
        self._end = None
        self._name = name
        self._total_duration = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._start = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        self._end = datetime.now()
        duration = (self._end - self._start).total_seconds()
        self._total_duration += duration
        logs[self._name] = duration
        logs["total_duration"] = self._total_duration
