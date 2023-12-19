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
# Modified   : Tuesday December 19th 2023 05:18:58 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Plotting Utilities"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
def suppress_ticks(axes: list) -> None:
    """Suppresses ticks on x and y axes."""
    if isinstance(axes, plt.Axes):
        _ = axes.set_xticks([])
        _ = axes.set_yticks([])
    else:
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
    title: str = None,
    fontsize_ax_title: int = 10,
    fontsize_fig_title: int = 12,
    fontsize_label: int = 10,
    show_ticks: bool = False,
    show: bool = False,
) -> plt.Figure:
    """Plots a list of images

    Args:
        images (list): List of images, each in numpy array format.
        nrows (int): Number of rows in plot. Number of columns is inferred from the number
            of images and the number of rows.
        titles (list): List of titles for each image. Used only for multiple images.
        title (str): The figure title.
        fontsize_ax_title (int): Font size for axes title. Default is 10. Used only for multiple images.
        fontsize_fig_title (int): Font size for figure title. Default is 12.
        fontsize_label (int): Font size for axes xlabel. Default is 10.
        suppress_ticks (bool): If True, x and y ticks are suppressed.
        show (bool): Whether to show the plot.
    """

    # ------------------------------------------------------------------------------------------- #
    #                                   VALIDATION                                                #
    # ------------------------------------------------------------------------------------------- #
    # Ensure images is a list
    if not isinstance(images, list):
        msg = "Images must be a list"
        logging.exception(msg)
        raise TypeError(msg)

    # Ensure that number of images is an integer multiple of the number of rows.
    if len(images) % nrows != 0:
        msg = "Number of images must be an integer multiple of the number of columns."
        logging.exception(msg)
        raise ValueError(msg)
    # Ensure titles match number of images
    if titles:
        if len(titles) != len(images):
            msg = f"Length of titles {len(titles)} must match number of images {len(images)}."
            logging.exception(msg)
            raise ValueError(msg)
    # ------------------------------------------------------------------------------------------- #

    n_images = len(images)
    xlabels = XLABELS[:n_images]

    ncols = int(n_images / nrows)
    height = nrows * 4
    width = min(ncols * 4, 12)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))

    if n_images == 1:
        _ = axes.imshow(images[0], cmap="gray", aspect="auto")
    else:
        axes = axes.flatten()

        for i, image in enumerate(images):
            _ = axes[i].imshow(image, cmap="gray", aspect="auto")
            if titles:
                _ = axes[i].set_title(label=titles[i], fontsize=fontsize_ax_title)
            _ = axes[i].set_xlabel(xlabel=xlabels[i], fontsize=fontsize_label)

    if title:
        _ = fig.suptitle(title, fontsize=fontsize_fig_title)

    if not show_ticks:
        suppress_ticks(axes)

    _ = plt.tight_layout(rect=(0, 0, 1, 0.99))

    if show:
        _ = plt.show()
    return fig


XLABELS = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(e)",
    "(f)",
    "(g)",
    "(h)",
    "(i)",
    "(j)",
    "(k)",
    "(l)",
    "(m)",
    "(n)",
    "(o)",
    "(p)",
    "(q)",
    "(r)",
    "(s)",
    "(t)",
    "(u)",
    "(v)",
    "(w)",
    "(x)",
    "(y)",
    "(z)",
]
