#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/utils/math.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday December 20th 2023 12:20:50 am                                            #
# Modified   : Wednesday December 20th 2023 01:04:22 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Math Module"""

import numpy as np


# ------------------------------------------------------------------------------------------------ #
def find_factors(c: int, non_prime_approx: bool = True) -> tuple[int, int]:
    """Calculate the closest two factors of c.

    The two factors of c that are closest; in other words, the
    closest two integers for which a*b=c. If c is a perfect square, the
    result will be [sqrt(c), sqrt(c)];

    if c is a prime number, the result will be [1, c].
    if non_prime_approx, factors are returned for the closest
    non-prime number.

    The first number will always be the smallest, if they
    are not equal.

    Args:
        c (int): The number for which the factors are to be found.
        non_prime_approx (bool): Whether to return approximate factors
            for prime numbers.

    Returns (int, int)
    """

    def _find_factors(c: int):
        if c // 1 != c:
            raise TypeError("c must be an integer.")

        a, b, i = 1, c, 0
        while a < b:
            i += 1
            if c % i == 0:
                a = i
                b = c // a

        return tuple(np.sort(np.array([a, b])))

    def _find_non_prime_appoximation(c: int):
        a = int(np.sqrt(c))
        b = c // a
        return tuple(np.sort(np.array([a, b])))

    f = _find_factors(c)
    if 1 in f and non_prime_approx:
        f = _find_non_prime_appoximation(c)
    return f
