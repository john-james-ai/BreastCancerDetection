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
# Modified   : Monday February 12th 2024 03:38:30 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from bcd import DataClass
from bcd.model.network.base import Network


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=unused-argument
# ================================================================================================ #
#                                  FINE TUNE SCHEDULE                                              #
# ================================================================================================ #
@dataclass
class FineTuneSchedule(DataClass):
    """Abstract base class for fine tuning schedules"""

    sessions: int = 5  # Number of fine tuning sessions
    patience: int = 2  # Early stopping patience for sessions.
    thaw_schedule: list = None  # Number of layers to thaw each session
    learning_rate_schedule: list = None  # Learning rates each session
    epochs: list = None  # Number of epochs to train each session.

    @abstractmethod
    def create(self, network: Network) -> None:
        """Creates the fine tuning schedule, including:
        - thaw_schedule: Rate at which layers are thawed each session.
        - learning_rate_schedule: Learning rates each session
        - epochs: Number of epochs to train each session.
        """

    def thaw(self, network: Network, session: int) -> Network:
        """Thaws the network according to the schedule

        Args:
            network (Network): Network to be fine tuned.
            session (int): The current session
        """
        n_layers = self.thaw_schedule[session]
        network.base_model.trainable = True
        for layer in network.base_model.layers[:-n_layers]:
            layer.trainable = False
        return network

    def update_learning_rate(
        self, session: int, optimizer: type[tf.keras.optimizers.Optimizer]
    ) -> tf.keras.optimizers.Optimizer:
        """Updates the learning rate on the optimizer

        Args:
            session (int): Current session
            optimizer (type[tf.keras.optimizers.Optimizer]): The optimizer class.
        """
        return optimizer(learning_rate=self.learning_rate_schedule[session])

    def get_epochs(self, session: int) -> int:
        return self.epochs[session]


# ------------------------------------------------------------------------------------------------ #
#                            LINEAR FINE TUNE SCHEDULE                                             #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class LinearFineTuneSchedule(FineTuneSchedule):
    """Creates a linear fine tune schedule.

    For this schedule, the number of layers to thaw grows linearly from 1 to the number of layers
    in the base model. Likewise, the number of epochs to train grows linearly from a epochs init
    to a epochs end for the number of sessions, subject to any early stopping.
    Finally, the learning rate decays linearly from a maximum to a minimum value over the
    number of sessions.
    """

    iceblocks: int = 10  # Number of partitions or blocks of layers to potentially thaw.
    learning_rate_init: float = 1e-5  #  The initial learning rate for the first session
    learning_rate_end: float = 1e-8  # The minimum learning rate over all sessions
    epochs_init: int = 5  # Number of epochs to train first session
    epochs_end: int = 50  # Number of epochs to train last session.

    def create_schedule(self, network: Network) -> None:
        # Get number of layers in base model.
        n_layers = len(network.base_model.layers)
        # Compute linear that schedule
        start = int(n_layers / self.sessions)
        # Untrimmed schedule is one that gradually unfreezes all layers.
        untrimmed_schedule = list(
            np.linspace(start=start, stop=n_layers, endpoint=True, num=self.sessions)
        )
        # Only the first 'sessions' blocks will be unfrozen.
        self.thaw_schedule = untrimmed_schedule[: self.sessions]

        # Compute learning rate schedule
        self.learning_rate_schedule = list(
            np.linspace(
                start=self.learning_rate_init,
                stop=self.learning_rate_end,
                endpoint=True,
                num=self.sessions,
            )
        )

        # Create epoch schedule
        self.epochs = list(
            np.linspace(
                start=self.epochs_init,
                stop=self.epochs_end,
                endpoint=True,
                num=self.sessions,
            )
        )


# ------------------------------------------------------------------------------------------------ #
#                              LOG FINE TUNE SCHEDULE                                              #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class LogFineTuneSchedule(FineTuneSchedule):
    """Creates a log fine tune schedule.

    For this schedule, the number of layers to thaw grows logarithmically from 1 to the number of layers
    in the base model. Likewise, the number of epochs to train grows logarithmically from epochs init
    to epochs end for the number of sessions, subject to any early stopping.
    Finally, the learning rate decays logarithmically from a maximum to a minimum value over the
    number of sessions.
    """

    iceblocks: int = 10  # Number of partitions or blocks of layers to potentially thaw.
    learning_rate_init: float = 1e-5  #  The initial learning rate for the first session
    learning_rate_end: float = 1e-8  # The minimum learning rate over all sessions
    epochs_init: int = 5  # Number of epochs to train first session
    epochs_end: int = 50  # Number of epochs to train last session.

    def create_schedule(self, network: Network) -> None:
        # Get number of layers in base model.
        n_layers = len(network.base_model.layers)
        # Compute linear schedule
        start = 1
        # Untrimmed schedule is one that gradually unfreezes all layers.
        untrimmed_schedule = list(
            np.geomspace(start=start, stop=n_layers, endpoint=True, num=self.iceblocks)
        )
        # Only the first 'sessions' blocks will be unfrozen.
        self.thaw_schedule = untrimmed_schedule[: self.sessions]

        # Compute learning rate schedule
        self.learning_rate_schedule = list(
            np.geomspace(
                start=self.learning_rate_init,
                stop=self.learning_rate_end,
                endpoint=True,
                num=self.sessions,
            )
        )

        # Create epoch schedule
        self.epochs = list(
            np.geomspace(
                start=self.epochs_init,
                stop=self.epochs_end,
                endpoint=True,
                num=self.sessions,
            )
        )


# ------------------------------------------------------------------------------------------------ #
#                                CUSTOM FINE TUNE SCHEDULE                                         #
# ------------------------------------------------------------------------------------------------ #
class CustomFineTuneSchedule(FineTuneSchedule):
    """Allows users to designated a custom fine tuning schedule

    The user provices:
        - sessions (int): Number of fine tuning sessions.
        - thaw schedule: List of layers to thaw each session.
        - learning_rate_schedule: List of learning rates for each session
        - epochs; List of epochs to train each session.
    """

    def create_schedule(self, network: Network) -> None:
        """User defined schedule."""
