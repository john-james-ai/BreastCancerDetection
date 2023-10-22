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
# Modified   : Sunday October 22nd 2023 12:02:43 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging

from dependency_injector import containers, providers

from bcd.manage_data.storage.mysql import MySQLDatabase
from bcd.manage_data.storage.config import DatabaseConfig
from bcd.manage_data.storage.repo import ImageRepo


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
#                                        DATA                                                      #
# ------------------------------------------------------------------------------------------------ #
class PersistenceContainer(containers.DeclarativeContainer):
    db = providers.Singleton(MySQLDatabase, config=DatabaseConfig)
    image_repo = providers.Singleton(ImageRepo)

    appdata_repo = providers.Singleton(AppDataRepo, database=db, config=FileConfig)
    review_repo = providers.Singleton(ReviewRepo, database=db, config=FileConfig)
    rating_repo = providers.Singleton(RatingRepo, database=db, config=FileConfig)
    job_repo = providers.Singleton(JobRepo, database=db, config=FileConfig)
    project_repo = providers.Singleton(AppDataProjectRepo, database=db, config=FileConfig)
    rating_jobrun_repo = providers.Singleton(RatingJobRunRepo, database=db, config=FileConfig)
    review_jobrun_repo = providers.Singleton(ReviewJobRunRepo, database=db, config=FileConfig)
    review_request_repo = providers.Singleton(ReviewRequestRepo, database=db, config=FileConfig)

    uow = providers.Singleton(
        UoW,
        database=db,
        appdata_repo=AppDataRepo,
        review_repo=ReviewRepo,
        rating_repo=RatingRepo,
        appdata_project_repo=AppDataProjectRepo,
        job_repo=JobRepo,
        rating_jobrun_repo=RatingJobRunRepo,
        review_jobrun_repo=ReviewJobRunRepo,
        review_request_repo=ReviewRequestRepo,
    )


# ------------------------------------------------------------------------------------------------ #
#                                       CONTAINER                                                  #
# ------------------------------------------------------------------------------------------------ #
class BCDContainer(containers.DeclarativeContainer):
    config = providers.Configuration(yaml_files=["config.yml"])

    logs = providers.Container(LoggingContainer, config=config)

    data = providers.Container(PersistenceContainer)
