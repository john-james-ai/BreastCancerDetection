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
# Modified   : Monday January 22nd 2024 03:03:28 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
import logging
import math
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
#                                LINEAR THAW SCHEDULE                                              #
# ------------------------------------------------------------------------------------------------ #
class LogThawSchedule(ThawSchedule):
    """Produces a logarithmic thaw schedule"""

    def __init__(
        self, sessions: int, base_model_layer: int, n_layers: int, start_layer: int = 1
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
#                                LEARNING RATE SCHEDULE FACTORY                                    #
# ------------------------------------------------------------------------------------------------ #
class LearningRateScheduleFactory(ABC):
    """Base Class for Learning Rate Schedule Factories."""

    @abstractmethod
    def __call__(self, session: int) -> float:
        """Returns the learning rate for the session."""


# ------------------------------------------------------------------------------------------------ #
#                               COSINE LEARNING RATE SCHEDULE                                      #
# ------------------------------------------------------------------------------------------------ #
tf_keras.saving.get_custom_objects().clear()


@tf_keras.saving.register_keras_serializable()
class BCDCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay with optional warmup.

    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.

    For the idea of a linear warmup of our learning rate,
    see [Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf).

    When we begin training a model, we often want an initial increase in our
    learning rate followed by a decay. If `warmup_target` is an int, this
    schedule applies a linear increase per optimizer step to our learning rate
    from `initial_learning_rate` to `warmup_target` for a duration of
    `warmup_steps`. Afterwards, it applies a cosine decay function taking our
    learning rate from `warmup_target` to `alpha` for a duration of
    `decay_steps`. If `warmup_target` is None we skip warmup and our decay
    will take our learning rate from `initial_learning_rate` to `alpha`.
    It requires a `step` value to  compute the learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a warmup followed by a
    decayed learning rate when passed the current optimizer step. This can be
    useful for changing the learning rate value across different invocations of
    optimizer functions.

    Our warmup is computed as:

    ```python
    def warmup_learning_rate(step):
        completed_fraction = step / warmup_steps
        total_delta = target_warmup - initial_learning_rate
        return completed_fraction * total_delta
    ```

    And our decay is computed as:

    ```python
    if warmup_target is None:
        initial_decay_lr = initial_learning_rate
    else:
        initial_decay_lr = warmup_target

    def decayed_learning_rate(step):
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_decay_lr * decayed
    ```

    Example usage without warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0.1
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    ```

    Example usage with warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0
    warmup_steps = 1000
    target_learning_rate = 0.1
    lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
    )
    ```

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        alpha=0.0,
        name=None,
        warmup_target=None,
        warmup_steps=0,
    ):
        """Applies cosine decay to the learning rate.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python int. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python int.
            Number of steps to decay over.
          alpha: A scalar `float32` or `float64` `Tensor` or a Python int.
            Minimum learning rate value for decay as a fraction of
            `initial_learning_rate`.
          name: String. Optional name of the operation.  Defaults to
            'CosineDecay'.
          warmup_target: None or a scalar `float32` or `float64` `Tensor` or a
            Python int. The target learning rate for our warmup phase. Will cast
            to the `initial_learning_rate` datatype. Setting to None will skip
            warmup and begins decay phase from `initial_learning_rate`.
            Otherwise scheduler will warmup from `initial_learning_rate` to
            `warmup_target`.
          warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python int.
            Number of steps to warmup over.
        """
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        self.warmup_steps = warmup_steps
        self.warmup_target = warmup_target
        self._model = None
        self.params = None

    def _decay_function(self, step, decay_steps, decay_from_lr, dtype):
        with tf.name_scope(self.name or "BCDCosineDecay"):
            completed_fraction = step / decay_steps
            tf_pi = tf.constant(math.pi, dtype=dtype)
            cosine_decayed = 0.5 * (1.0 + tf.cos(tf_pi * completed_fraction))
            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return tf.multiply(decay_from_lr, decayed)

    def _warmup_function(
        self, step, warmup_steps, warmup_target, initial_learning_rate
    ):
        with tf.name_scope(self.name or "BCDCosineDecay"):
            completed_fraction = step / warmup_steps
            total_step_delta = warmup_target - initial_learning_rate
            dtype = completed_fraction.dtype
            total_step_delta = tf.cast(total_step_delta, dtype)
            initial_learning_rate = tf.cast(initial_learning_rate, dtype)
            return total_step_delta * completed_fraction + initial_learning_rate

    def __call__(self, step):
        with tf.name_scope(self.name or "BCDCosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            global_step_recomp = tf.cast(step, dtype)

            if self.warmup_target is None:
                global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
                return self._decay_function(
                    global_step_recomp,
                    decay_steps,
                    initial_learning_rate,
                    dtype,
                )

            warmup_target = tf.cast(self.warmup_target, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)

            global_step_recomp = tf.minimum(
                global_step_recomp, decay_steps + warmup_steps
            )

            return tf.cond(
                global_step_recomp < warmup_steps,
                lambda: self._warmup_function(
                    global_step_recomp,
                    warmup_steps,
                    warmup_target,
                    initial_learning_rate,
                ),
                lambda: self._decay_function(
                    global_step_recomp - warmup_steps,
                    decay_steps,
                    warmup_target,
                    dtype,
                ),
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
        }


# ------------------------------------------------------------------------------------------------ #
#                             COSINE LEARNING RATE SCHEDULE FACTORY                                #
# ------------------------------------------------------------------------------------------------ #
class CosineDecayLearningRateFactory(LearningRateScheduleFactory):
    """Cosine decay learning rate factory

    Args:
        sessions (int): Number of fine tuning sessions
        epochs (int): Number of epochs to decay over
        batches (int): Number of batches in the training set
        warmup_phase (float): The proportion of the total number of steps used to warm up.

    """

    def __init__(
        self,
        sessions: int,
        epochs: int,
        batches: int,
        min_lr: float = 1e-10,
        max_lr: float = 1e-4,
        warmup_phase: float = 0.5,
    ) -> None:
        self._sessions = sessions
        self._epochs = epochs
        self._batches = batches
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._total_steps = epochs * batches
        self._warmup_steps = int(warmup_phase * self._total_steps)
        self._decay_steps = self._total_steps - self._warmup_steps
        self._schedule = None

    def __call__(
        self, session: int
    ) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        return BCDCosineDecay(
            initial_learning_rate=0,
            decay_steps=self._decay_steps,
            warmup_target=self.schedule[session - 1],
            warmup_steps=self._warmup_steps,
        )

    @property
    def schedule(self) -> list:
        """Creates schedule of target learning rates"""
        if self._schedule is None:
            self._schedule = list(
                np.linspace(
                    start=self._max_lr,
                    stop=self._min_lr,
                    num=self._sessions,
                    endpoint=True,
                )
            )
        return self._schedule
