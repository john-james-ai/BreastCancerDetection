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
# Created    : Saturday February 17th 2024 11:21:47 pm                                             #
# Modified   : Sunday February 18th 2024 03:53:05 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import math

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=unused-argument
# ------------------------------------------------------------------------------------------------ #
#                               LEARNING RATE RANGE TEST                                           #
# ------------------------------------------------------------------------------------------------ #
class LRRangeTestCallback(tf.keras.callbacks.Callback):
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
        self._logger.setLevel(logging.INFO)

    def on_epoch_begin(self, epoch, logs=None) -> None:
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


# ------------------------------------------------------------------------------------------------ #
#                            TRIANGLE LEARNING RATE SCHEDULE                                       #
# ------------------------------------------------------------------------------------------------ #
class TriangleLearningRateScheduleCallback(tf.keras.callbacks.Callback):
    """Creates Triangle Learning Rate Schedule

    The triangle learning rate policy is a cyclical learning rate schedule in which the learning
    rate changes linearly from a minimum to a maximum learning rate and back over a number of
    epochs called a cycle. [1_]

    References
    [1] L. N. Smith, “Cyclical Learning Rates for Training Neural Networks.” arXiv, Apr. 04, 2017.
    Accessed: Feb. 17, 2024. [Online]. Available: http://arxiv.org/abs/1506.01186

    """

    def __init__(
        self, min_lr: float = 1e-5, max_lr: float = 0.1, step_size: int = 4
    ) -> None:
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._step_size = step_size
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(logging.INFO)

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        cycle = math.floor(1 + epoch / (2 * self._step_size))
        x = np.abs(epoch / self._step_size - 2 * cycle + 1)
        lr = self._min_lr + (self._max_lr - self._min_lr) * np.max(0, (1 - x))
        msg = f"Setting learning rate to {lr}."
        self._logger.info(msg)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
