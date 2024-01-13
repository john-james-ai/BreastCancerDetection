#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/utils/model.py                                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday January 13th 2024 04:09:00 pm                                              #
# Modified   : Saturday January 13th 2024 05:05:39 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2024 John James                                                                 #
# ================================================================================================ #
"""Model Utilities"""
import tensorflow as tf


# ------------------------------------------------------------------------------------------------ #
def thaw(n: int, model: tf.keras.Model, base_model_layer: int) -> tf.keras.Model:
    """Thaws n top layers of a TensorFlow model

    Used for fine-tuning in transfer learning. This assumes a TensorFlow model including
    a base model with a designated trained architecture. Only the layers of the
    underlying base model are thawed. Thus the layer number for the base model must be
    provided.

    Args:
        n (int): Number of top layers to thaw.
        model (tf.keras.Model): TensorFlow Model
        base_model_layer (int): Index for the layer containing the base model

    """
    # Thaw the entire model.
    model.layers[base_model_layer].trainable = True
    for layer in model.layers[base_model_layer].layers[:-n]:
        layer.trainable = False

    return model
