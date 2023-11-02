#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/flow/builder.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday November 1st 2023 02:00:57 am                                             #
# Modified   : Wednesday November 1st 2023 08:20:55 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging

from bcd.dal.repo.uow import UoW
from bcd.preprocess.image.flow.basetask import Task
from bcd.preprocess.image.method.basemethod import Method, Param


# ------------------------------------------------------------------------------------------------ #
class TaskBuilder:
    """Abstract base class for Builder"""

    def __init__(self, uow: UoW) -> None:
        self._uow = uow
        self._task = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    def task(self) -> Task:
        """Returns the complete operation object"""
        self.validate()
        task = self._task
        self.reset()
        return task

    def reset(self) -> None:
        """Resets the Task object."""
        self._task = None

    def set_task(self, task: type[Task]) -> None:
        """Adds the Task class to the builder.

        Args:
            task (type[Task]): A Task subclass type.
        """
        if not isinstance(task(), Task):
            msg = "task is not a valid Task object."
            self._logger.exception(msg)
            raise TypeError(msg)

        self._task = task()
        self._task.uow = self._uow

    def add_task_params(self, params: Param) -> None:
        """Adds the method params to the object

        Args:
            params (Param): Parameters that control Task behavior.
        """
        if not isinstance(params, Param):
            msg = "task_params is not a valid Param object."
            self._logger.exception(msg)
            raise TypeError(msg)
        self._task.task_params = params

    def add_method(self, method: Method) -> None:
        """Adds the underlying method to the object.

        Args:
            method (Method): A Method instance.
        """
        if not isinstance(method, Method):
            msg = "method is not a valid Method object."
            self._logger.exception(msg)
            raise TypeError(msg)
        self._task.method = method

    def add_method_params(self, params: Param) -> None:
        """Adds the method params to the object

        Args:
            params (Param): Parameters that control method behavior.
        """
        if not isinstance(params, Param):
            msg = "method_params is not a valid Param object."
            self._logger.exception(msg)
            raise TypeError(msg)
        self._task.method_params = params

    def validate(self) -> None:
        is_valid = False
        if self._task is None:
            msg = "The Task object cannot be None."
        elif self._task.method is None:
            msg = "Task Method must be not be None."
        else:
            is_valid = True
        if not is_valid:
            self._logger.exception(msg)
            raise TypeError(msg)

        return (
            self._task is not None
            and self._task.method is not None
            and self._task.method_params is not None
            and self._task.uow is not None
        )
