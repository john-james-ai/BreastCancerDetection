#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/architecture.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday February 10th 2024 07:11:33 am                                             #
# Modified   : Saturday February 10th 2024 10:41:16 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""CNN Architecture Module"""
from abc import ABC, abstractmethod

import tensorflow as tf

# ------------------------------------------------------------------------------------------------ #


class Architecture(ABC):
    """Abstract base class for CNN architectures

    Args:
        base_model (tf.keras.Model): The base model upon which the architecture is built.
        output_shape (int): Shape of output. Default = 1
        activation (str): Activation function providing class probabilities. Default = 'sigmoid'
    """
