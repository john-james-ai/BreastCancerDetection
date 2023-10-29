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
# Modified   : Sunday October 29th 2023 02:19:55 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
import logging.config

from dependency_injector import containers, providers

from bcd.config import Config
from bcd.core.image.factory import ImageFactory
from bcd.dal.database.mysql import MySQLDatabase
from bcd.dal.io.image import ImageIO
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

    db_config = providers.Singleton(Config)

    db = providers.Singleton(MySQLDatabase, config=db_config)

    io = providers.Singleton(ImageIO)

    image_factory = providers.Singleton(
        ImageFactory,
        metadata_filepath=config.data.metadata,
        mode=config.mode,
        directory=config.data[config.mode],
        io=io,
    )

    image_repo = providers.Singleton(
        ImageRepo, database=db, image_factory=image_factory, io=io, mode=config.mode
    )

    task_repo = providers.Singleton(TaskRepo, database=db, mode=config.mode)

    uow = providers.Singleton(
        UoW,
        database=db,
        image_factory=image_factory,
        image_repo=ImageRepo,
        task_repo=TaskRepo,
        io=io,
        mode=config.mode,
    )


# ------------------------------------------------------------------------------------------------ #
#                                       CONTAINER                                                  #
# ------------------------------------------------------------------------------------------------ #
class BCDContainer(containers.DeclarativeContainer):
    """Dependency Injection Container"""

    config = providers.Configuration(yaml_files=["config.yml"])

    logs = providers.Container(LoggingContainer, config=config)

    dal = providers.Container(DALContainer, config=config)
