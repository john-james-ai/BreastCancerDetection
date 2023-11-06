#!/usr/bin/bash
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.12                                                                             #
# Filename   : /scripts/database/command.sh                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 8th 2023 09:52:11 am                                                 #
# Modified   : Sunday November 5th 2023 11:18:04 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #

echo $'\nRunning Database Commands...'
sudo mysql -u root -p --database mysql < scripts/database/image_table.sql
echo $'\nDatabase Commands Complete.'