#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/base_model.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday November 5th 2023 09:20:17 am                                                #
# Modified   : Sunday November 5th 2023 11:27:01 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base Module for Breast Cancer Classification"""
from keras import layers

from bcd.model.base import BCDModel


# ------------------------------------------------------------------------------------------------ #
# pylint: disable=arguments-differ
# ------------------------------------------------------------------------------------------------ #
class BaseModel(BCDModel):
    """Base Model"""

    def __init__(self, img_height: int, img_width: int, num_classes: int):
        super().__init__()
        self._img_height = img_height
        self._img_width = img_width
        self._num_classes = num_classes
        self.resizing = layers.Resizing(height=img_height, width=img_width)
        self.rescaling = layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3))
        self.conv1 = layers.Conv2D(16, 3, padding="same", activation="relu")
        self.maxpooling1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(32, 3, padding="same", activation="relu")
        self.maxpooling2 = layers.MaxPooling2D()
        self.conv3 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.maxpooling3 = layers.MaxPooling2D()
        self.dropout = layers.Dropout(0.2)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(128, activation="relu")
        self.classification = layers.Dense(num_classes)

    def call(self, x):
        x = self.resizing(x)
        x = self.rescaling(x)
        x = self.conv1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.maxpooling2(x)
        x = self.conv3(x)
        x = self.maxpooling3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.classification(x)

    def get_config(self) -> dict:
        return {
            "img_height": self._img_height,
            "img_width": self._img_width,
            "num_classes": self._num_classes,
        }
