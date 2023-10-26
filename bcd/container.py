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
# Modified   : Thursday October 26th 2023 01:31:58 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
import logging.config

from dependency_injector import containers, providers

from bcd.infrastructure.database.mysql import MySQLDatabase
from bcd.infrastructure.database.config import DatabaseConfig
from bcd.core.image.repo import ImageRepo
from bcd.config import Config

# from bcd.manage_data.repo.task import TaskRepo
from bcd.core.image.factory import ImageFactory


# ------------------------------------------------------------------------------------------------ #
#                                        LOGGING                                                   #
# ------------------------------------------------------------------------------------------------ #
class LoggingContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )


# ------------------------------------------------------------------------------------------------ #
#                                        REPO                                                      #
# ------------------------------------------------------------------------------------------------ #
class RepoContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    db = providers.Singleton(MySQLDatabase, config=DatabaseConfig)

    factory = providers.Singleton(ImageFactory, case_fp=config.image.factory.case_fp, config=Config)

    image = providers.Singleton(ImageRepo, database=db, image_factory=factory, config=Config)

    # task = providers.Factory(TaskRepo, database=db, mode=config.mode)


# ------------------------------------------------------------------------------------------------ #
#                                       CONTAINER                                                  #
# ------------------------------------------------------------------------------------------------ #
class BCDContainer(containers.DeclarativeContainer):
    config = providers.Configuration(yaml_files=["config.yml"])

    logs = providers.Container(LoggingContainer, config=config)

    repo = providers.Container(RepoContainer, config=config)


if __name__ == "__main__":
    container = BCDContainer()
    container.init_resources()
    container.wire(
        packages=["bcd.core", "bcd.preprocess.convert", "bcd.preprocess.filter", "bcd.core"]
    )
