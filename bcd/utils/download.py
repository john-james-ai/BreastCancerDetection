#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/utils/download.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday December 27th 2023 06:38:39 pm                                            #
# Modified   : Thursday December 28th 2023 04:26:03 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Download Managwer"""
import asyncio
import logging
import os

import aiofile
import aiohttp
import requests
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# ------------------------------------------------------------------------------------------------ #


class Downloader:
    """Download Manager Class

    Args:
        force (bool): If False, local data will be overwritten by data downloaded. Otherwise,
                an exception will be raised if the destination directory or
                filepath exists. Default is False
        progressbar (bool): Whether to log the progress of the download. Default is False

    """

    def __init__(self, force: bool = False, progressbar: bool = True) -> None:
        self._force = force
        self._progressbar = progressbar
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def download_file(self, url: str, destination: str, chunksize: int = 1024) -> None:
        """Downloads a file from the specified url

        Args:
            url (str): The download link to the resource
            destination (str): Local destination directory or filepath
            chunksize (int): Size of chunks in bytes. None will stream
                data as it arrives.

        """
        destination = self._format_filepath(url=url, destination=destination)
        proceed = self._check_filepath(filepath=destination)
        if proceed:
            self._download_sync(url=url, destination=destination, chunksize=chunksize)

    def download_package(
        self,
        urls: list,
        destination: str,
        parallel: bool = True,
        semaphore: int = 5,
        chunksize: int = 1024,
    ) -> None:
        """Asynchronously downloads a package of data to a destination directory

        Args:
            urls (list): A list of URLs.
            destination (str): Download destination directory or filename. Directory must
                have a trailing backslash. Destinations with no backslash will be
                treated as file paths.
            parallel (bool): Whether to download synchronously, or asynchronously (parallel). Default is True
            semaphore (int): Only used if parallel is True. It limits the number
                of simultaneous downloads. Default is 5.
            chunksize (int): Size of chunks in bytes. Default = 1024. Used only when parallel
                is False.


        """
        proceed = self._check_directory(directory=destination)
        if proceed:
            if parallel:
                self._download_async(
                    urls=urls, destination=destination, semaphore=semaphore
                )
            else:
                for url in urls:
                    self.download_file(
                        url=url, destination=destination, chunksize=chunksize
                    )

    def _download_async(self, urls: list, destination: str, semaphore: int = 5):
        """Downloads a list of URLs asynchronously to a destination directory."""

        sema = asyncio.BoundedSemaphore(semaphore)

        async def fetch_file(session, url):
            filename = os.path.basename(url)
            async with sema:
                self._logger.debug(url)
                async with session.get(url) as resp:
                    assert resp.status == 200
                    data = await resp.read()

            async with aiofile.async_open(
                os.path.join(destination, filename), "wb"
            ) as file:
                await file.write(data)

        async def download():
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_file(session, url) for url in urls]
                if self._progressbar:
                    await tqdm_asyncio.gather(*tasks)
                else:
                    await asyncio.gather(*tasks)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(download())
        loop.close()

    def _download_sync(self, url: str, destination: str, chunksize: int = 1024) -> None:
        """Downloads using the requests module"""
        with requests.Session() as session:
            resp = session.get(url, stream=True)
            resp.raise_for_status()
            if self._progressbar:
                total = int(resp.headers.get("content-length", 0))
                with open(destination, "wb") as file, tqdm(
                    desc=os.path.basename(destination),
                    total=total,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=chunksize,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=chunksize):
                        size = file.write(chunk)
                        bar.update(size)

            else:
                with open(destination, "wb") as file:
                    for chunk in resp.iter_content(chunk_size=chunksize):
                        file.write(chunk)

    def _format_filepath(self, url: str, destination: str) -> str:
        """Ensures the destination is a file path

        If destination is a directory, the filename from the URL
        is appended to the destination. If the destination
        is a filename, it is not changed.

        """
        if destination[-1] == os.sep:
            filename = os.path.basename(url)
            destination = os.path.join(destination, filename)
        return destination

    def _check_filepath(self, filepath: str) -> bool:
        """Checks file existence and handles vis-a-vis force."""

        if os.path.exists(filepath) and not self._force:
            self._raise_file_exists_warning(filepath=filepath)
            return False
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            return True

    def _check_directory(self, directory: str) -> bool:
        """Checks existence of directory and handles vis-a-vis force."""
        if os.path.exists(directory) and not self._force:
            self._raise_file_exists_warning(filepath=directory)
            return False
        else:
            os.makedirs(directory, exist_ok=True)
            return True

    def _raise_file_exists_warning(self, filepath: str) -> None:
        msg = f"{filepath} already exists. If you wish to download and overwrite existing data, set force to True."
        self._logger.warning(msg)
