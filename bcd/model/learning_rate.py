#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/learning_rate.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 17th 2024 11:21:47 pm                                             #
# Modified   : Saturday February 17th 2024 11:42:40 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=unused-argument
# ------------------------------------------------------------------------------------------------ #
class LRRangeTest(tf.keras.callbacks.Callback):
    """Increases learning rate linearly over a number of epochs

    Args:
        min_lr (float): Minimum learning rate. Default is 1e-4.
        max_lr (float): Maximum learning rate. Default is 1e-1,
        epochs (int): Number of epochs for the learning rate range test. Default is 10.
    """

    def __init__(
        self, min_lr: float = 1e-4, max_lr: float = 1e-1, epochs: int = 10
    ) -> None:
        super().__init__()
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._epochs = epochs
        self._learning_rates = self._build_lr_schedule()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def on_epoch_end(self, epoch, logs=None) -> None:
        idx = epoch % self._epochs
        lr = self._learning_rates[idx]
        msg = f"Setting learning rate to {lr}."
        self._logger.info(msg)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def _build_lr_schedule(self) -> list:
        return list(
            np.linspace(
                start=self._min_lr, stop=self._max_lr, endpoint=True, num=self._epochs
            )
        )
