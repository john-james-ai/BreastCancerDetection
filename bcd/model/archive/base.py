#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/model/base.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday November 5th 2023 11:02:05 am                                                #
# Modified   : Saturday December 30th 2023 05:05:43 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# pylint: disable=import-error
# ------------------------------------------------------------------------------------------------ #
import os
from datetime import datetime

from keras import Model

from bcd.config import Config


# ------------------------------------------------------------------------------------------------ #
class BCDModel(Model):
    """Base Model extending keras.Model definition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        now = datetime.now()
        self._name = (
            self.__class__.__name__.lower() + "_" + now.strftime("%Y-%m-%d_%H-%M-%S")
        )

    @property
    def name(self) -> str:
        return self._name

    def call(self, inputs, training=None, mask=None):
        """Calls the model on the new inputs and returns the outputs as tensors."""

    def get_config(self):
        """Returns the config of the `Model`."""

    def save_model(self, overwrite: bool = True, **kwargs) -> None:
        directory = Config.get_model_dir()
        os.makedirs(directory, exist_ok=True)
        filename = self.name + ".keras"
        filepath = os.path.join(directory, filename)
        self.save(filepath=filepath, overwrite=overwrite, save_format="tf", **kwargs)
