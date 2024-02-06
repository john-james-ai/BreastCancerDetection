#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/schedule.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday January 21st 2024 03:44:39 pm                                                #
# Modified   : Monday January 22nd 2024 01:37:40 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=unused-argument
# ------------------------------------------------------------------------------------------------ #
#                                   THAW SCHEDULE                                                  #
# ------------------------------------------------------------------------------------------------ #
class ThawSchedule(ABC):
    """Base class for Thaw Schedule classes.

    Thaws the top n layers of a model based upon the schedule defined in the subclass

    Args:
        sessions (int): Total number of fine tuning sessions
        base_model_layer (int): Layer containing the pre-trained model
        n_layers (int): Number of layers in the pre-trained model.
    """

    def __init__(
        self, sessions: int, base_model_layer: int, n_layers: int, **kwargs
    ) -> None:
        self._sessions = sessions
        self._base_model_layer = base_model_layer
        self._n_layers = n_layers
        self._schedule = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(level=logging.DEBUG)

    def __call__(self, model: tf.keras.Model, session: int) -> tf.keras.Model:
        """Performs the thawing

        Args:
            model (tf.keras.Model): Model containing the pre-trained model
            session (int): Current session. Sessions start at 1
        """
        n = self.schedule[session - 1]
        model.layers[self._base_model_layer].trainable = True
        for layer in model.layers[self._base_model_layer].layers[:-n]:
            layer.trainable = False

        print("\n")
        msg = f"Thawed {n} layers of the base model."
        self._logger.info(msg)

        return model

    @property
    @abstractmethod
    def schedule(self) -> list:
        """Computes the thaw schedule"""


# ------------------------------------------------------------------------------------------------ #
#                                LINEAR THAW SCHEDULE                                              #
# ------------------------------------------------------------------------------------------------ #
class LinearThawSchedule(ThawSchedule):
    """Produces a linear thaw schedule"""

    @property
    def schedule(self) -> list:
        if self._schedule is None:
            start = int(self._n_layers / self._sessions)
            self._schedule = list(
                np.linspace(
                    start=start, stop=self._n_layers, endpoint=True, num=self._sessions
                ).astype(int)
            )
        return self._schedule


# ------------------------------------------------------------------------------------------------ #
#                                 LOG THAW SCHEDULE                                                #
# ------------------------------------------------------------------------------------------------ #
class LogThawSchedule(ThawSchedule):
    """Produces a logarithmic thaw schedule"""

    def __init__(
        self, sessions: int, base_model_layer: int, n_layers: int, start_layer: int = 2
    ) -> None:
        super().__init__(
            sessions=sessions, base_model_layer=base_model_layer, n_layers=n_layers
        )
        self._start_layer = start_layer

    @property
    def schedule(self) -> list:
        if self._schedule is None:
            self._schedule = list(
                np.geomspace(
                    start=self._start_layer,
                    stop=self._n_layers,
                    endpoint=True,
                    num=self._sessions,
                ).astype(int)
            )
        return self._schedule


# ------------------------------------------------------------------------------------------------ #
#                                   LEARNING RATE SCHEDULE                                         #
# ------------------------------------------------------------------------------------------------ #
class LearningRateSchedule(ABC):
    """Base class for transfer learning session learning rate schedule classes.

    Subclasses define schedules for adjusting the learning rates between transfer
    learning sessions rather than epochs or steps.

    Args:
        sessions (int): Total number of fine tuning sessions
        initial_learning_rate (float): Learning rate for first session.
        final_learning_rate (float): Learning rate for final session.
    """

    def __init__(
        self,
        sessions: int = 10,
        initial_learning_rate: float = 1e-4,
        final_learning_rate: float = 1e-9,
    ) -> None:
        self._sessions = sessions
        self._initial_learning_rate = initial_learning_rate
        self._final_learning_rate = final_learning_rate
        self._schedule = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(level=logging.DEBUG)

    def __call__(self, session: int) -> tf.keras.Model:
        """Performs the thawing

        Args:
            session (int): Current session. Sessions start at 1
        """
        print("\n")
        msg = f"Set learning rate to {self.schedule[session-1]}"
        self._logger.info(msg)
        return self.schedule[session - 1]

    @property
    @abstractmethod
    def schedule(self) -> list:
        """Computes the thaw schedule"""


# ------------------------------------------------------------------------------------------------ #
#                                LINEAR LEARNING RATE                                              #
# ------------------------------------------------------------------------------------------------ #
class LinearLearningRateSchedule(LearningRateSchedule):
    """Produces a linear learning rate schedule"""

    @property
    def schedule(self) -> list:
        if self._schedule is None:
            self._schedule = list(
                np.linspace(
                    start=self._initial_learning_rate,
                    stop=self._final_learning_rate,
                    endpoint=True,
                    num=self._sessions,
                )
            )
        return self._schedule


# ------------------------------------------------------------------------------------------------ #
#                              LOG LEARNING RATE SCHEDULE                                          #
# ------------------------------------------------------------------------------------------------ #
class LogLearningRateSchedule(LearningRateSchedule):
    """Produces a logarithmic learning rate schedule"""

    @property
    def schedule(self) -> list:
        if self._schedule is None:
            self._schedule = list(
                np.geomspace(
                    start=self._initial_learning_rate,
                    stop=self._final_learning_rate,
                    endpoint=True,
                    num=self._sessions,
                )
            )
        return self._schedule
