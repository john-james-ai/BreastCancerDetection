#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/container.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 21st 2023 07:43:26 pm                                              #
# Modified   : Wednesday November 1st 2023 01:55:01 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
import logging.config

from dependency_injector import containers, providers

from bcd.dal.database.mysql import MySQLDatabase
from bcd.dal.repo.evaluation import EvalRepo
from bcd.dal.repo.image import ImageRepo
from bcd.dal.repo.task import TaskRepo
from bcd.dal.repo.uow import UoW

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=c-extension-no-member
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
#                                        LOGGING                                                   #
# ------------------------------------------------------------------------------------------------ #
class LoggingContainer(containers.DeclarativeContainer):
    """Contains logging resource"""

    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )


# ------------------------------------------------------------------------------------------------ #
#                                        REPO                                                      #
# ------------------------------------------------------------------------------------------------ #
class DALContainer(containers.DeclarativeContainer):
    """Contains the Data Access Layer"""

    config = providers.Configuration()

    db = providers.Singleton(MySQLDatabase)

    image_repo = providers.Singleton(ImageRepo, database=db)

    task_repo = providers.Singleton(TaskRepo, database=db)

    eval_repo = providers.Singleton(EvalRepo, database=db)

    uow = providers.Singleton(
        UoW,
        database=db,
        image_repo=ImageRepo,
        task_repo=TaskRepo,
        eval_repo=EvalRepo,
    )


# ------------------------------------------------------------------------------------------------ #
#                                       CONTAINER                                                  #
# ------------------------------------------------------------------------------------------------ #
class BCDContainer(containers.DeclarativeContainer):
    """Dependency Injection Container"""

    config = providers.Configuration(yaml_files=["config.yml"])

    logs = providers.Container(LoggingContainer, config=config)

    dal = providers.Container(DALContainer, config=config)
