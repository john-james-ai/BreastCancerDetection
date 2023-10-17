#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/prep/image/base.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 16th 2023 08:41:25 pm                                                #
# Modified   : Tuesday October 17th 2023 06:56:56 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Image Preparation Base Module"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------------ #
class DICOMTask(ABC):
    """Defines interface for DICOM preprocessing tasks."""

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Executes the task on the input and returns the task output."""

    def display(self, img: np.array, cmap: str = "jet", cbar: bool = True) -> None:
        """Displays a single mammogram image

        Args:
            img (np.array): Image pixel data in numpy array format.
            cmap (str): Color map. Default is 'jet'
            cbar (bool): Indicates whether a color bar is to be displayed.
        """
        plt.imshow(img, cmap=cmap)
        if cbar:
            plt.colorbar()
        plt.show()
