#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/utils/object.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 28th 2023 03:15:26 pm                                              #
# Modified   : Wednesday November 1st 2023 11:20:07 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import importlib
import logging
from typing import Callable


# ------------------------------------------------------------------------------------------------ #
def get_class(class_name: str, module_name: str) -> type[Callable]:
    """Converts a string to a class instance."""

    try:
        module = importlib.import_module(module_name)
        try:
            class_ = getattr(module, class_name)
        except AttributeError:
            logging.exception("Class does not exist")
            raise
    except ImportError:
        logging.exception("Module does not exist")
        raise
    return class_ or None
