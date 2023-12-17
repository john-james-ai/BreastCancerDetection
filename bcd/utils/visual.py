#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/utils/visual.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday December 17th 2023 03:18:15 pm                                               #
# Modified   : Sunday December 17th 2023 05:13:02 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Plotting Utilities"""
import logging

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------------------------------------ #
def suppress_ticks(axes: list) -> None:
    """Suppresses ticks on x and y axes."""
    for ax in axes:
        _ = ax.set_xticks([])
        _ = ax.set_yticks([])


# ------------------------------------------------------------------------------------------------ #
#                                    PLOT IMAGES                                                   #
# ------------------------------------------------------------------------------------------------ #
def plot_images(
    images: list,
    nrows: int = 1,
    titles: list = None,
    labels: list = None,
    title: str = None,
    fontsize_ax_title: int = 10,
    fontsize_fig_title: int = 12,
    fontsize_label: int = 10,
    show: bool = False,
) -> plt.Figure:
    """Plots a list of images

    Args:
        images (list): List of images, each in numpy array format.
        nrows (int): Number of rows in plot. Number of columns is inferred from the number
            of images and the number of rows.
        titles (list): List of titles for each image.
        labels (list): List x axis labels for each image.
        title (str): The figure title.
        fontsize_ax_title (int): Font size for axes title. Default is 10.
        fontsize_fig_title (int): Font size for figure title. Default is 12.
        fontsize_label (int): Font size for axes xlabel. Default is 10.
        show (bool): Whether to show the plot.
    """
    n_images = len(images)
    # ------------------------------------------------------------------------------------------- #
    #                                   VALIDATION                                                #
    # ------------------------------------------------------------------------------------------- #
    # Ensure that number of images is an integer multiple of the number of rows.
    if n_images % nrows != 0:
        msg = "Number of images must be an integer multiple of the number of columns."
        logging.exception(msg)
        raise ValueError(msg)
    # Ensure titles match number of images
    if titles:
        if len(titles) != len(images):
            msg = f"Length of titles {len(titles)} must match number of images {len(images)}."
            logging.exception(msg)
            raise ValueError(msg)

    # Ensure labels match number of images
    if labels:
        if len(labels) != len(images):
            msg = f"Length of labels {len(labels)} must match number of images {len(images)}."
            logging.exception(msg)
            raise ValueError(msg)

    # ------------------------------------------------------------------------------------------- #

    ncols = int(n_images / nrows)
    height = nrows * 4
    width = 12

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
    axes = axes.flatten()

    for i, image in enumerate(images):
        _ = axes[i].imshow(image, cmap="gray", aspect="auto")
        if titles:
            _ = axes[i].set_title(label=titles[i], fontsize=fontsize_ax_title)
        if labels:
            _ = axes[i].set_xlabel(xlabel=labels[i], fontsize=fontsize_label)

    if title:
        _ = fig.suptitle(title, fontsize=fontsize_fig_title)

    suppress_ticks(axes)
    _ = plt.tight_layout(rect=(0, 0, 1, 0.99))

    if show:
        _ = plt.show()
    return fig
