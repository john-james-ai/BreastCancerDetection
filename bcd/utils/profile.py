#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/utils/profile.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday December 25th 2023 04:14:05 pm                                               #
# Modified   : Monday December 25th 2023 06:01:29 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Profiling Decorator Module"""
import functools
import logging
import time
import tracemalloc
from logging.handlers import TimedRotatingFileHandler

import logmatic

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=no-member
# ------------------------------------------------------------------------------------------------ #
LOGFILE = "log/profile/log"
logger = logging.getLogger(__name__)
fh = TimedRotatingFileHandler(filename=LOGFILE, when="D", interval=1, encoding="utf-8")
fh.setFormatter(logmatic.JsonFormatter())
logger.addHandler(fh)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------------------------------------------ #
def profiler(func):
    """Profiles the runtime performance and resource utilization of the decorated function."""

    @functools.wraps(func)
    def profiler_wrapper(*args, **kwargs):
        # Obtain name of class and method
        qualname = func.__qualname__

        # Start the clocks
        st_wall = time.time()
        st_cpu = time.process_time()
        tracemalloc.start()

        # Execute the wrapped method
        result = func(*args, **kwargs)

        # Capture wall and cpu time, and memory
        et_wall = time.time()
        et_cpu = time.process_time()
        mem = tracemalloc.get_traced_memory()
        wall_time = et_wall - st_wall
        cpu_time = et_cpu - st_cpu

        # Format the data and log
        profile = {
            "Qualname": qualname,
            "time": wall_time,
            "cpu time": cpu_time,
            "memory": mem,
        }
        logger.info(msg="Profiling", extra=profile)

        return result

    return profiler_wrapper
