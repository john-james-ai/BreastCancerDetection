#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/decorator.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday November 1st 2023 09:14:37 am                                             #
# Modified   : Wednesday November 1st 2023 02:54:31 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import functools


# ------------------------------------------------------------------------------------------------ #
#                                 TIMER DECORATOR                                                  #
# ------------------------------------------------------------------------------------------------ #
def timer(func):
    """Wrapper for Task subclass run methods that automatically calls the start and stop methods."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.start()
        result = func(self, *args, **kwargs)
        self.stop()
        return result

    return wrapper


# ------------------------------------------------------------------------------------------------ #
#                                 COUNTER DECORATOR                                                #
# ------------------------------------------------------------------------------------------------ #
def counter(func):
    """Wrapper for Task subclass process_image method that automatically counts images processed."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.count()
        return result

    return wrapper
